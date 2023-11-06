import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tb
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.utils.clip_grad import clip_grad_norm_

from TaCoGPT.inference.bleu import BLEU
from TaCoGPT.model.layers import ModelArgs
from TaCoGPT.train.finetune import accuracy_idx
from TaCoGPT.train.warmup import GradualWarmupScheduler


class TrainArgs:
    fasta2lineage_file_path: str  # meta information
    train_data_folder: str
    test_data_folder: str
    test_sample_time_per_genome: int
    model_weight_save_folder: str
    model_weight_load_path: str
    lineage_vocab_path: str
    lineage_n: int = 6
    seq_max_len: int = 4096
    kmer_vocab_path: str
    tree_idx_path: str
    kmer_k: int = 3
    train_epoch: int = 98
    train_repeat_time_per_epoch: int = 4
    batch_size: int = 4
    regu_lambda: float = 5e-4
    lr: float = 1e-6
    lr_multiple: float = 10
    lr_warmup_epoch: int = 15
    loss_state: str = "mean"  # mean or sum
    loss_gamma: float = 1.0
    search_algo: str
    topk: int = 3
    beam_width: int = 3
    beam_batch_size: int = 4
    eval_epoch: int = 10
    trainOReval: str = "train"


def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params


def pretrain_GPT(
    model: nn.Module,
    training_param: TrainArgs,
    model_param: ModelArgs,
    train_dist_loader,
    test_loader,
    device,
    local_rank=-1,
) -> None:
    if local_rank is None:
        local_rank = dist.get_rank()
    timeNow = str(time.asctime()).replace(" ", "_").replace(";", "").replace(":", "_")
    writer = tb.writer.SummaryWriter("./log/" + timeNow + "/")
    optimizer = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=optim.AdamW,
        lr=training_param.lr,
        weight_decay=training_param.regu_lambda,
        eps=1e-8,
        betas=(0.9, 0.95))
    warmUpScheduler = GradualWarmupScheduler(
        optimizer,
        training_param.lr_multiple,
        training_param.lr_warmup_epoch,
        training_param.train_epoch - training_param.lr_warmup_epoch + 1,
    )
    loss_func_ce = nn.CrossEntropyLoss(
        reduction=training_param.loss_state, ignore_index=model_param.pad_id, label_smoothing=0.001).to(device)
    scaler = GradScaler()
    trainingStep = 0
    for e in range(1, training_param.train_epoch + 1):
        # repeat N times for current epoch
        for r in range(training_param.train_repeat_time_per_epoch):
            train_dist_loader.sampler.set_epoch(e + r)
            model.train()
            n_batches = len(train_dist_loader) * training_param.train_repeat_time_per_epoch
            data_len = len(train_dist_loader)
            losses = 0.0
            cur_train_step = 0
            for i, batch_seqs in enumerate(train_dist_loader):
                idx = r * data_len + i + 1
                inputs = batch_seqs.to(device)
                targets = inputs[:, 1:].contiguous()
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    lm_logits = model(inputs)[:, :-1, :]  # [b, seqlen, c]
                    lm_logits = lm_logits.contiguous().view(-1, model_param.k_vocab_size)
                    targets = targets.view(-1)
                    loss = loss_func_ce(lm_logits, targets)
                loss_val = loss.item()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.module.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                # loss.backward()
                # optimizer.step()
                trainingStep += 1
                cur_train_step += 1
                losses += loss_val
                if local_rank in [-1, 0]:
                    writer.add_scalar("Loss with training step", loss_val, trainingStep)
                print("[RANK: {}] | Epoch {} Iteration({}/{}) | Loss: {:.4f} Loss: {:.4f} | LR: {:.8f}".format(
                    local_rank,
                    e,
                    idx,
                    n_batches,
                    losses / cur_train_step,
                    loss_val,
                    optimizer.param_groups[0]["lr"],
                )
                )
                if trainingStep % 10 == 0:
                    with torch.no_grad():
                        p_argmax = torch.argmax(lm_logits, dim=-1)[0:7].tolist()
                        t_t = targets[0:7].tolist()
                        print("[RANK: {}] | pre_g: {} truth_g: {}".format(
                            local_rank, p_argmax, t_t,))
        # record losses
        if local_rank in [-1, 0]:
            writer.add_scalar("Loss with epoch", losses / n_batches, e)
        if local_rank in [-1, 0]:
            if e % training_param.eval_epoch == 0 or e == training_param.train_epoch:
                with torch.no_grad():
                    l, bleu_score = test_GPT(model.module, training_param, model_param,
                                             test_loader, e, device, writer)

                ckpt_name = "Epoch_{}".format(
                    e) + "-" + "loss_{:.4f}".format(l) + "-" + "bleu_{:.4f}".format(bleu_score) + ".pth"
                torch.save(model.module.state_dict(), os.path.join(
                    training_param.model_weight_save_folder, ckpt_name))
        dist.barrier()
        warmUpScheduler.step()


def test_GPT(
    model,
    training_param: TrainArgs,
    model_param: ModelArgs,
    test_dataloader,
    e,
    device,
    tf_writer=None,
):
    n_batches = len(test_dataloader)
    loss_s = 0
    model.eval()
    loss_func_ce = nn.CrossEntropyLoss(
        reduction=training_param.loss_state, ignore_index=model_param.pad_id).to(device)
    print("Eval start.")
    candidates = []
    references = []
    with torch.no_grad():
        for i, batch_seqs in enumerate(test_dataloader):
            inputs = batch_seqs.to(device)
            targets = inputs[:, 1:].contiguous()
            lm_logits = model(inputs)[:, :-1, :]  # [b, seqlen, c]
            lm_logits_arg_max = torch.argmax(lm_logits, dim=-1)  # [b, seqlen-1]
            for cand_seq, ref_seq in zip(lm_logits_arg_max, targets):
                cur_index = torch.nonzero(ref_seq, as_tuple=True)[0][-1].item()
                candidates.append(list(map(lambda x: str(x), cand_seq[0: cur_index + 1].tolist())))
                references.append(list(map(lambda x: str(x), ref_seq[0: cur_index + 1].tolist())))
            lm_logits = lm_logits.contiguous().view(-1, model_param.k_vocab_size)
            targets = targets.view(-1)
            loss = loss_func_ce(lm_logits, targets)
            loss_val = loss.item()
            loss_s += loss_val
            print("[RANK: {}] Epoch {} Iteration({}/{}) loss: {:.4f} loss_cur: {:.4f}".format(
                device, e, i, n_batches, loss_s / (i + 1.0), loss_val))
    print("Eval Epoch {} Loss: {:.4f}".format(e, loss_s / n_batches))
    bleu_score = BLEU(candidates, references, n_gram=3)
    print("BLEU SCORE IS: {}".format(bleu_score))
    if tf_writer is not None:
        tf_writer.add_scalar("Eval loss per epoch", loss_s / n_batches + 0.0, e)
        tf_writer.add_scalar("BLEU Score per epoch", bleu_score, e)
    return loss_s / n_batches, bleu_score


def pretrain_TaCoGPT(
    model: nn.Module,
    training_param: TrainArgs,
    model_param: ModelArgs,
    train_dist_loader,
    test_loader,
    device,
    local_rank=-1,
) -> None:
    if local_rank is None:
        local_rank = dist.get_rank()

    timeNow = str(time.asctime()).replace(" ", "_").replace(";", "").replace(":", "_")
    writer = tb.writer.SummaryWriter("./log/" + timeNow + "/")
    optimizer = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=optim.AdamW,
        lr=training_param.lr,
        weight_decay=training_param.regu_lambda,
        betas=(0.9, 0.95),
    )
    warmUpScheduler = GradualWarmupScheduler(
        optimizer,
        training_param.lr_multiple,
        training_param.lr_warmup_epoch,
        training_param.train_epoch - training_param.lr_warmup_epoch + 1,
    )
    loss_func_bce = nn.BCEWithLogitsLoss(reduction=training_param.loss_state)

    trainingStep = 0
    for e in range(1, training_param.train_epoch + 1):
        # repeat N times for current epoch
        loss2_weight = 5.0
        for r in range(training_param.train_repeat_time_per_epoch):
            model.train()
            n_batches = len(train_dist_loader) * training_param.train_repeat_time_per_epoch
            data_len = len(train_dist_loader)
            losses = 0.0
            cur_train_step = 0

            for i, (batch_seq1, batch_seq2, batch_label) in enumerate(train_dist_loader):
                idx = r * data_len + i + 1
                batch_seq1 = batch_seq1.to(device)
                batch_seq2 = batch_seq2.to(device)
                batch_label = batch_label.to(device)

                batch_inputs = torch.cat([batch_seq1, batch_seq2], dim=0)
                optimizer.zero_grad(set_to_none=True)

                pred_logit = model(batch_inputs)
                loss = loss_func_bce(pred_logit, batch_label)

                loss_val = loss.item()
                losses += loss_val

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.module.parameters(), 1.0)
                optimizer.step()
                trainingStep += 1
                cur_train_step += 1
                acc_contra = accuracy_idx(pred_logit, batch_label, True).item()
                writer.add_scalar("Loss with training steps", loss_val, trainingStep)
                writer.add_scalar("Accuracy with training steps", acc_contra, trainingStep)
                print("[RANK: {}] | Epoch {} Iteration({}/{}) | Loss: {:.4f} Loss_cur: {:.4f} | ACC_contra: {:.4f} | LR: {:.8f}".format(
                    local_rank,
                    e,
                    idx,
                    n_batches,
                    losses / cur_train_step,
                    loss_val,
                    acc_contra,
                    optimizer.param_groups[0]["lr"],
                )
                )
                if trainingStep % 10 == 0:
                    with torch.no_grad():
                        con_argmax = torch.sigmoid(pred_logit)[0:7].tolist()
                        t_c = batch_label[0:7].tolist()
                        print("[RANK: {}] | pre_c: {} truth_c: {}".format(
                            local_rank, con_argmax, t_c))

        # record losses
        if local_rank in [-1, 0]:
            writer.add_scalar("Loss with epoch", losses / n_batches, e)

        ckpt_name = None
        if local_rank in [-1, 0] and (e % training_param.eval_epoch == 0 or e == training_param.train_epoch):
            with torch.no_grad():
                l, a = pretrain_TaCoGPT_test(model.module, training_param,
                                             model_param, test_loader, e, device, writer, None)
            ckpt_name = "Epoch_{}".format(e) + "-" + "loss_{:.2f}".format(-l) + \
                "-" + "ACC_{:.4f}".format(a) + ".pth"
        # test part
        if local_rank in [-1, 0] and ckpt_name is not None:
            torch.save(model.module.state_dict(), os.path.join(
                training_param.model_weight_save_folder, ckpt_name))

        dist.barrier()
        warmUpScheduler.step()
        loss2_weight *= 0.995
    writer.close()


def pretrain_TaCoGPT_test(
    model,
    training_param: TrainArgs,
    model_param: ModelArgs,
    test_dataloader,
    e,
    device,
    tf_writer=None,
    output_path=None,
):
    n_batches = len(test_dataloader)
    loss_s = 0
    acc_s = 0
    wh = None
    if output_path is not None:
        wh = open(output_path, "a")
    model.eval()
    loss_func_bce = nn.BCEWithLogitsLoss(reduction=training_param.loss_state)
    print("Eval start.")
    with torch.no_grad():
        for i, (batch_seq1, batch_seq2, batch_label) in enumerate(test_dataloader):
            batch_seq1 = batch_seq1.to(device)
            batch_seq2 = batch_seq2.to(device)
            batch_label = batch_label.to(device)

            batch_inputs = torch.cat([batch_seq1, batch_seq2], dim=0)

            pred_logit = model(batch_inputs)
            loss = loss_func_bce(pred_logit, batch_label)

            loss_val = loss.item()
            acc_contra = accuracy_idx(pred_logit, batch_label, True).item()

            acc_s += acc_contra
            loss_s += loss_val
            print("[RANK: {}] Epoch {} Iteration({}/{}) loss: {:.4f} loss_cur: {:.4f} ACC_c: {}".format(
                device, e, i, n_batches, loss_s / (i + 1.0), loss_val, acc_contra)
            )

    if tf_writer is not None:
        tf_writer.add_scalar("Eval loss per epoch", loss_s / n_batches + 0.0, e)
        tf_writer.add_scalar("Eval Accuracy", acc_s / n_batches + 0.0, e)
    print("Eval Epoch {} Loss: {:.4f} Acc_c: {:.5f}".format(
        e, loss_s / n_batches, acc_s / n_batches))
    if wh is not None:
        wh.close()
    return loss_s / n_batches, acc_s / n_batches
