import os
import time
from typing import Callable, Dict, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tb
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.distributed import all_gather, get_world_size
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader

from TaCoGPT.data.utils_data import convert_indices2seq
from TaCoGPT.inference.bleu import BLEU
from TaCoGPT.inference.utils_infer import beam_search, topk_search, top_k_top_p_filtering
from TaCoGPT.IO import readPickle, readVocabulary
from TaCoGPT.model.layers import ModelArgs, empty_cache
from TaCoGPT.model.Loss import FocalCrossEntropyLoss
from TaCoGPT.train.warmup import GradualWarmupScheduler
from TaCoGPT.utils_func import rank_accuracy_calculate_from_test_output_file


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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k / batch_size)
    return res


def accuracy_idx(pre_idx, target, binary=False):
    if binary:
        pre_idx = (torch.sigmoid(pre_idx) >= 0.5).float()
    correct = pre_idx.eq(target)
    correct_k = correct.contiguous().view(-1).float().mean()
    return correct_k


def taxonomic_rank_acc_stat(pre_idx: torch.Tensor, target: torch.Tensor, num_rank=6):
    correct_info = [0. for _ in range(num_rank)]
    count_info = [0. for _ in range(num_rank)]
    for i, (p, t) in enumerate(zip(pre_idx, target)):
        idx = i % num_rank
        if p.equal(t):
            correct_info[idx] += 1
        count_info[idx] += 1
    correct_info = np.array(correct_info, dtype=np.float32) / np.array(count_info, dtype=np.float32)
    return correct_info


def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params


def requires_grad_change(names: list, model: nn.Module):
    for key, val in model.named_parameters():
        for name in names:
            if name in key:
                val.requires_grad = False
                val.requires_grad_(False)


def train_TaCoGPT(
    model: nn.Module,
    training_param: TrainArgs,
    model_param: ModelArgs,
    train_dist_loader,
    test_loader,
    device,
    local_rank=-1,
    gpip=False,
    res_out_path=None,
    eval_mode=False
    # # gpt mode
    # generating_seq: bool = False,
    # gpt_model: nn.Module = None,
    # gpt_kmer_k: int = 4,
    # gpt_kmer2index: Dict[str, int] = None,
    # beam_width :int= 2,
    # prompt_len: int = 800,
    # generating_len: int = 100
):
    """function to train LLaMa

    Args:
        model (nn.Module): LLaMa model.
        training_param (TrainArgs): training parameters.
        train_dist_loader (_type_): training dataloader, distributed loader.
        test_loader (_type_): testing dataloader, not distributed loader
    """
    if local_rank is None:
        local_rank = dist.get_rank()
    with torch.no_grad():
        weight = torch.Tensor([1.5, 1.4, 1.3, 1.2, 1.2, 1.2, 0.1]).float().to(device)
    tree_idx = readPickle(training_param.tree_idx_path)
    timeNow = str(time.asctime()).replace(" ", "_").replace(";", "").replace(":", "_")
    writer = tb.writer.SummaryWriter("./log/" + timeNow + "/")
    if gpip is False:
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=optim.AdamW,
            lr=training_param.lr,
            weight_decay=training_param.regu_lambda,
            betas=(0.9, 0.99),
            eps=1e-5)
    else:
        optimizer = optim.AdamW(model.parameters(),
                                lr=training_param.lr,
                                weight_decay=training_param.regu_lambda,
                                betas=(0.9, 0.99),
                                eps=1e-5)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer,
        training_param.lr_multiple,
        training_param.lr_warmup_epoch,
        training_param.train_epoch - training_param.lr_warmup_epoch + 1,
    )
    loss_func_fce = FocalCrossEntropyLoss(reduction=training_param.loss_state,
                                          gamma=training_param.loss_gamma).to(device)
    trainingStep = 0
    scaler = GradScaler()
    losses = 0.0
    n_batches = 1.

    for e in range(1, training_param.train_epoch + 1):
        # repeat N times for current epoch
        for r in range(training_param.train_repeat_time_per_epoch):
            if eval_mode:
                break
            if gpip is False:
                train_dist_loader.sampler.set_epoch(e + r)
            model.train()
            n_batches = len(train_dist_loader) * training_param.train_repeat_time_per_epoch
            data_len = len(train_dist_loader)
            losses = 0.0
            cur_train_step = 0
            for i, (batch_seqs, batch_lineages, batch_phy, b_contrast_lineages, b_contrast_label) in enumerate(
                train_dist_loader
            ):
                idx = r * data_len + i + 1
                if gpip is False:
                    batch_seqs = batch_seqs.to(device)
                    batch_lineages = batch_lineages.to(device)
                    b_contrast_lineages = b_contrast_lineages.to(device)
                else:
                    batch_seqs = batch_seqs.to("cuda:0")
                    batch_lineages = batch_lineages.to("cuda:0")
                    b_contrast_lineages = b_contrast_lineages.to("cuda:0")
                b = batch_seqs.size(0)
                weight_f_loss = weight.expand(size=[b, 7]).contiguous().view(-1)
                batch_phy = batch_phy.to(device)
                b_contrast_label = b_contrast_label.to(device)
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    X = torch.cat([batch_seqs, batch_lineages], dim=-1)
                    if gpip is False:
                        batch_model_output, predict_output_phy, predict_output_contrast = model(
                            X, b_contrast_lineages
                        )  # B, L2-1, C_wordsvocab
                    else:
                        batch_model_output, predict_output_phy, predict_output_contrast = model(
                            X, b_contrast_lineages
                        ).local_value()  # B, L2-1, C_wordsvocab
                    # print(batch_model_output)
                    batch_lineages = batch_lineages.to(device)
                    targets = batch_lineages[:, 1:].contiguous().view(-1)
                    predict_output = batch_model_output.contiguous().view(-1, model_param.a_vocab_size)
                    phy_targets = batch_phy.contiguous().view(-1)
                    contrast_targets = b_contrast_label.contiguous().view(-1)
                    loss1 = loss_func_fce(predict_output, targets, weight_f_loss, "sum") / (b * 6.0)
                    loss2 = loss_func_fce(predict_output_phy, phy_targets)
                    loss3 = loss_func_fce(predict_output_contrast, contrast_targets)
                    loss = loss1 + loss2 + loss3
                    loss1_val = loss1.item()
                    loss2_val = loss2.item()
                    loss3_val = loss3.item()
                    loss_val = loss.item()
                    losses += loss_val
                scaler.scale(loss).backward()
                # check gradients
                for para in model.parameters():
                    if para.requires_grad is True and para.grad.isnan().float().sum() != 0:
                        para.grad = torch.zeros_like(para.grad).float().to(device)
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                # loss.backward()
                # optimizer.step()
                trainingStep += 1
                cur_train_step += 1
                acc = accuracy(predict_output, targets)[0].item()
                acc_phy = accuracy(predict_output_phy, phy_targets)[0].item()
                acc_contra = accuracy(predict_output_contrast, contrast_targets)[0].item()
                writer.add_scalar("Loss with training steps", loss_val, trainingStep)
                writer.add_scalar("Accuracy with training steps", acc, trainingStep)
                print(
                    "[RANK: {}] | Epoch {} Iteration({}/{}) | Loss: {:.4f} Loss_1: {:.4f} Loss_2: {:.4f} Loss_3: {:.4f} | ACC_g: {:.4f} ACC_phy: {:.4f} ACC_contra: {:.4f} | LR: {:.8f}".format(
                        local_rank,
                        e,
                        idx,
                        n_batches,
                        losses / cur_train_step,
                        loss1_val,
                        loss2_val,
                        loss3_val,
                        acc,
                        acc_phy,
                        acc_contra,
                        optimizer.param_groups[0]["lr"],
                    )
                )
                if trainingStep % 10 == 0:
                    with torch.no_grad():
                        p_argmax = torch.argmax(predict_output, dim=-1)[0:7].tolist()
                        t_t = targets[0:7].tolist()

                        phy_argmax = torch.argmax(predict_output_phy, dim=-1)[0:7].tolist()
                        t_p = phy_targets[0:7].tolist()

                        con_argmax = torch.argmax(predict_output_contrast, dim=-1)[0:7].tolist()
                        t_c = contrast_targets[0:7].tolist()
                        print("[RANK: {}] | pre_g: {} truth_g: {}".format(
                            local_rank, p_argmax, t_t,))
                        print("[RANK: {}] | pre_p: {} truth_p: {}".format(
                            local_rank, phy_argmax, t_p))
                        print("[RANK: {}] | pre_c: {} truth_c: {}".format(
                            local_rank, con_argmax, t_c))
        # record losses
        if local_rank in [-1, 0]:
            writer.add_scalar("Loss with epoch", losses / n_batches, e)

        # test part
        assert res_out_path is not None, ValueError(f"res_out_path parameter is {res_out_path}.")
        ckpt_name = None
        if eval_mode or e % training_param.eval_epoch == 0 or e == training_param.train_epoch:
            if gpip:
                l, a, correct_info = test_TaCoGPT(model, training_param, model_param, test_loader,
                                                  e, device, tree_idx, writer, None, gpip, False)
            else:
                # l, a, correct_info = test_TaCoGPT(model.module, training_param, model_param,
                #                           test_loader, e, device, tree_idx, writer, None, gpip, generating_seq,
                #                           gpt_model, gpt_kmer_k, gpt_kmer2index, beam_width, prompt_len,
                #                           generating_len)
                if res_out_path[-1] != "/":
                    res_out_path += "/"
                res_out_path_cur = res_out_path + "all_"
                l, a, correct_info = test_TaCoGPT(model.module, training_param, model_param,
                                                  test_loader, e, device, tree_idx, writer, res_out_path_cur + str(local_rank) + ".txt", gpip, False)
            if gpip is False:
                dist.barrier()

            if local_rank in [-1, 0]:
                t_ranks = dist.get_world_size()
                zero_wh = open(res_out_path_cur + str(0) + ".txt", 'a')
                for r in range(1, t_ranks):
                    read_path = res_out_path_cur + str(r) + ".txt"
                    with open(read_path, 'r') as rh:
                        for line in rh:
                            zero_wh.write(line)
                zero_wh.close()
                acc_all = rank_accuracy_calculate_from_test_output_file(
                    res_out_path_cur + str(0) + ".txt", 6)
                ckpt_name = "Epoch_{}".format(
                    e) + "-rank-{:.2f}-{:.2f}-{:.2f}-{:.2f}-{:.2f}-{:.2f}".format(*acc_all) + ".pth"

            if local_rank in [-1, 0] and eval_mode is False:
                if gpip:
                    torch.save(model.state_dict(), os.path.join(
                        training_param.model_weight_save_folder, ckpt_name))
                else:
                    torch.save(model.module.state_dict(), os.path.join(
                        training_param.model_weight_save_folder, ckpt_name))
        dist.barrier()
        if local_rank in [-1, 0] and eval_mode is False:
            if ckpt_name is None:
                ckpt_name = f"Epoch-{e}"
            torch.save(
                model.module.state_dict(),
                os.path.join(
                    training_param.model_weight_save_folder, ckpt_name
                ),
            )
        if gpip is False:
            dist.barrier()
        if eval_mode:
            break
        else:
            warmUpScheduler.step()
    writer.close()


def test_TaCoGPT(
    model: nn.Module,
    training_param: TrainArgs,
    model_param: ModelArgs,
    test_dataloader: DataLoader,
    e,
    device,
    tree=None,
    tf_writer=None,
    output_path=None,
    gpip=False,
    # gpt model
    generating_seq_mode: bool = False,
    gpt_model: nn.Module = None,
    gpt_kmer_k: int = 4,
    gpt_kmer2index: Dict[str, int] = None,
    beam_width: int = 2,
    prompt_len: int = 800,
    generating_len: int = 100,
    beam_searchORtop_k: str = "top_k",
    top_k: int = 5,
    top_p: float = 0.99
):
    n_batches = len(test_dataloader) + 0.
    prob_s = 0
    acc_s = 0
    wh = None
    if output_path is not None:
        wh = open(output_path, "w")
    model.eval()

    test_dataset = test_dataloader.dataset

    correct_info = np.zeros(shape=[training_param.lineage_n], dtype=np.float32)
    with torch.no_grad():
        for i, (batch_seqs, batch_lineages, _, seqs) in enumerate(test_dataloader):
            if gpip is False:
                # seqs is a list of seq string.
                if generating_seq_mode:
                    assert gpt_model is not None, ValueError("GPT model is None.")
                    gpt_model = gpt_model.to(device)
                    base_gap = 1
                    gpt_index2kmer = {}
                    for key, val in gpt_kmer2index.items():
                        gpt_index2kmer[int(val)] = key
                    # data prepare
                    prompt_seqs = []
                    prev_seqs = []
                    p2p = {}
                    for j, seq in enumerate(seqs):
                        sub_seq = seq[-prompt_len:]
                        if len(sub_seq) != prompt_len:
                            print(
                                f"Warning, seq length is {len(sub_seq)}, not equal with the prompt length: {prompt_len}. We would omit it.")
                            p2p[j] = None
                        else:
                            indices = []
                            for k in range(0, len(sub_seq) - gpt_kmer_k + 1, base_gap):
                                nts = sub_seq[k: k + gpt_kmer_k]
                                if nts in gpt_kmer2index:
                                    indices.append(gpt_kmer2index[nts])
                                else:
                                    indices.append(gpt_kmer2index["<unk>"])
                            indices_tensor = torch.tensor(
                                indices, dtype=torch.long).unsqueeze(0).to(device)
                            prompt_seqs.append(indices_tensor)
                            prev_seqs.append(seq[0: -prompt_len])
                            p2p[j] = len(prompt_seqs) - 1
                    # generating, generated_seq: [b, beam_width, prompt_len + generating_len]
                    generated_seq = torch.cat(prompt_seqs, dim=0)
                    if beam_searchORtop_k == "beam_search":
                        generated_seq, _ = beam_search(
                            gpt_model, generated_seq, generating_len, beam_width)
                        generated_seq = generated_seq[:, 0, :]  # get the max log-prob
                    elif beam_searchORtop_k == "top_k":
                        for l in range(generating_len):
                            if (l + 1) % 20 == 0:
                                print(f"generating at {l + 1}, total {generating_len}")
                            next_logits = gpt_model(generated_seq)[:, -1, :]
                            pred_token = top_k_top_p_filtering(next_logits, top_k, top_p)
                            generated_seq = torch.cat([generated_seq, pred_token], dim=-1)
                    else:
                        ValueError("No implement other searching algos than top_k and beam_search.")

                    seqs_string_list = convert_indices2seq(generated_seq, gpt_index2kmer, 1)
                    batch_seqs_indices = []
                    for key, val in p2p.items():
                        if val is None:
                            batch_seqs_indices.append(batch_seqs[key].unsqueeze(0))
                        else:
                            cur_seq = prev_seqs[val] + seqs_string_list[val]
                            # print("generated: ", cur_seq)
                            # print("original", seqs[val])
                            batch_seqs_indices.append(
                                test_dataset.convert_seq2indices(cur_seq)[0].unsqueeze(0))
                    batch_seqs = torch.cat(batch_seqs_indices, dim=0).to(device)
                else:
                    batch_seqs = batch_seqs.to(device)
                b = batch_seqs.size(0)
                start = torch.ones(size=[b, 1], dtype=torch.long, device=device)
                x = torch.cat((batch_seqs, start), dim=-1)
                ori_len = x.size(-1)
            else:
                # gpip model would not be used later, then it would not apply the generateing DNA sequence mode.
                batch_seqs = batch_seqs.to("cuda:0")
                b = batch_seqs.size(0)
                start = torch.ones(size=[b, 1], dtype=torch.long, device="cuda:0")
                x = torch.cat((batch_seqs, start), dim=-1)
                ori_len = x.size(-1)
            assert ori_len == training_param.seq_max_len + 1, \
                f"ERROR WITH GENERATING... The length of input indices is {ori_len}, but it should be {training_param.seq_max_len + 1}"

            predict_out = None
            if training_param.search_algo == "topk":
                assert tree is not None
                predict_out = []
                prob_b = []
                for j in range(b):
                    # if j % 4 == 0:
                    #     print(
                    #         f"[RNAK: {device}] | Topk search at {j}-th item in this batch, total items in this batch: {b}")
                    empty_cache(model)
                    cur_x = x[j].unsqueeze(0)
                    predict_idxs_list, predict_probs_list, seq_rep_norm = topk_search(
                        model, cur_x, model_param.lineage_n, training_param.topk, tree, gpip)
                    prob_multi = []
                    for k, prob_list in enumerate(predict_probs_list):
                        ip = 1.0
                        for p in prob_list:
                            ip = ip * p
                        prob_multi.append((ip, predict_idxs_list[k], prob_list))
                    sorted_info = list(sorted(prob_multi, key=lambda x: x[0], reverse=True))[0:10]
                    candidate = []
                    candidate_ip = []
                    for ip, predict_idxs, _ in sorted_info:
                        candidate.append(predict_idxs)
                        candidate_ip.append(ip)
                    candidate_idxs_tensor = torch.tensor(
                        candidate, dtype=torch.long, device=device)  # [can_num, l]
                    can_num = len(candidate)
                    if gpip is False:
                        lineages_rep = model.lineage_encoder_norm(candidate_idxs_tensor)
                        lineages_rep = lineages_rep.view(1, can_num, model_param.dim)
                        logit_scale = model.logit_scale.exp()
                        predict_contrast = model.gatherValues(
                            seq_rep_norm, lineages_rep) * logit_scale  # [1, can_num]
                    else:
                        mds = [*model]
                        last_md = mds[-1]
                        first_md = mds[0]
                        candidate_idxs_tensor = candidate_idxs_tensor.to("cuda:0")
                        lineages_rep = first_md.lineage_encoder_norm(candidate_idxs_tensor)
                        lineages_rep = lineages_rep.view(1, can_num, model_param.dim).to(device)
                        logit_scale = last_md.logit_scale.exp()
                        predict_contrast = last_md.gatherValues(
                            seq_rep_norm, lineages_rep) * logit_scale
                    candidate_ip_tensor = torch.tensor(candidate_ip, dtype=torch.float32,
                                                       device=logit_scale.device).softmax(-1)
                    sim = torch.softmax(predict_contrast, dim=-1).squeeze(0) * 0.6 + \
                        candidate_ip_tensor * 0.4
                    # [can_num]
                    _, idx = sim.topk(k=1, dim=-1)
                    max_predict = sorted_info[idx.item()]
                    predict_out.append(max_predict[1])
                    prob_b.append(max_predict[0])
                    # info out to file
                    if output_path is not None:
                        cur_tar = batch_lineages[j][1:-1].tolist()
                        for k, ele in enumerate(max_predict[1]):
                            if k != (len(max_predict[1]) - 1):
                                wh.write(str(ele) + ",")
                            else:
                                wh.write(str(ele) + "\t")
                        for k, ele in enumerate(cur_tar):
                            if k != (len(cur_tar) - 1):
                                wh.write(str(ele) + ",")
                            else:
                                wh.write(str(ele) + "\t")
                        for k, ele in enumerate(max_predict[2]):
                            if k != (len(max_predict[2]) - 1):
                                wh.write(str(ele) + ",")
                            else:
                                wh.write(str(ele) + "\n")
                predict_out = torch.tensor(predict_out, dtype=torch.long,
                                           device=device).contiguous().view(-1)
            else:
                raise ValueError("No implement other seaching algos than Top-K.")
            # eval
            batch_lineages = batch_lineages.to(device)
            targets = batch_lineages[:, 1:-1].contiguous().view(-1)
            # print("predict_out shape", predict_out.shape)
            # print("target shape", targets.shape)
            acc = accuracy_idx(predict_out, targets).item()
            correct_rank_info = taxonomic_rank_acc_stat(
                predict_out, targets, num_rank=training_param.lineage_n)
            acc_s += acc
            cur_prob = sum(prob_b) / b + 0.0
            prob_s += cur_prob
            correct_info += correct_rank_info
            p_argmax = predict_out[0:7].tolist()
            t_t = targets[0:7].tolist()
            print("[RNAK: {}] | Epoch {} Iteration({}/{}) prob_mean: {:.4f} prob_cur: {:.4f} ACC: {:.4f} pre: {} truth: {}".format(
                device, e, i, n_batches, prob_s / (i + 1.0), cur_prob, acc, p_argmax, t_t))
            print("[RNAK: {}] | Curent ACC : p__{:4f}, c__{:4f}, o__{:4f}, f__{:4f}, g__{:4f}, s__{:4f}".format(
                device, *correct_rank_info))
            print("[RNAK: {}] | Mean ACC : p__{:4f}, c__{:4f}, o__{:4f}, f__{:4f}, g__{:4f}, s__{:4f}".format(device,
                                                                                                              *(correct_info / (i + 1.0))))
    if tf_writer is not None:
        tf_writer.add_scalar("Eval prob mean per epoch", prob_s / n_batches + 0.0, e)
        tf_writer.add_scalar("Eval Accuracy", acc_s / n_batches + 0.0, e)
    print("[RNAK: {}] | Eval Epoch {} prob_mean: {:.4f} Acc: {:.5f}%".format(
        device, e, prob_s / n_batches, acc_s / n_batches))
    res = correct_info / n_batches
    print("[RNAK: {}] | Final ACC in each rank: p__{:4f}, c__{:4f}, o__{:4f}, f__{:4f}, g__{:4f}, s__{:4f}".format(device, *res))
    if wh is not None:
        wh.close()
    return prob_s / n_batches, acc_s / n_batches, res


def train_TaCoGPT_remove_mode(
    model: nn.Module,
    training_param: TrainArgs,
    model_param: ModelArgs,
    train_dist_loader,
    test_loader_family,
    test_loader_genus,
    device,
    local_rank=-1,
    gpip=False,
    res_out_path=None,
    eval_mode=False
):
    """function to train LLaMa

    Args:
        model (nn.Module): LLaMa model.
        training_param (TrainArgs): training parameters.
        train_dist_loader (_type_): training dataloader, distributed loader.
        test_loader (_type_): testing dataloader, not distributed loader
    """
    if local_rank is None:
        local_rank = dist.get_rank()
    with torch.no_grad():
        weight = torch.Tensor([1.5, 1.4, 1.3, 1.2, 1.2, 1.2, 0.1]).float().to(device)
    tree_idx = readPickle(training_param.tree_idx_path)
    timeNow = str(time.asctime()).replace(" ", "_").replace(";", "").replace(":", "_")
    writer = tb.writer.SummaryWriter("./log/" + timeNow + "/")
    if gpip is False:
        lstm_feature_params = []
        others = []
        for name, val in model.named_parameters():
            if "lstm" in name or "feature" in name:
                lstm_feature_params.append(val)
            else:
                others.append(val)
        optimizer = ZeroRedundancyOptimizer(
            [{"params": lstm_feature_params, "lr": training_param.lr * 0.75},
             {"params": others}],
            optimizer_class=optim.AdamW,
            lr=training_param.lr,
            weight_decay=training_param.regu_lambda,
            betas=(0.9, 0.99),
            eps=1e-5
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_param.lr,
            weight_decay=training_param.regu_lambda,
            betas=(0.9, 0.99),
            eps=1e-5
        )
    warmUpScheduler = GradualWarmupScheduler(
        optimizer,
        training_param.lr_multiple,
        training_param.lr_warmup_epoch,
        training_param.train_epoch - training_param.lr_warmup_epoch + 1,
    )
    loss_func_fce = FocalCrossEntropyLoss(reduction=training_param.loss_state,
                                          gamma=training_param.loss_gamma).to(device)
    trainingStep = 0
    scaler = GradScaler()
    losses = 0.0
    n_batches = 1.0

    for e in range(1, training_param.train_epoch + 1):
        # repeat N times for current epoch
        for r in range(training_param.train_repeat_time_per_epoch):
            if eval_mode:
                break
            if gpip is False:
                train_dist_loader.sampler.set_epoch(e + r)
            model.train()
            n_batches = len(train_dist_loader) * training_param.train_repeat_time_per_epoch
            data_len = len(train_dist_loader)
            for i, (
                batch_seqs,
                batch_lineages,
                batch_phy,
                b_contrast_lineages,
                b_contrast_label,
            ) in enumerate(train_dist_loader):
                idx = r * data_len + i + 1
                if gpip is False:
                    batch_seqs = batch_seqs.to(device)
                    batch_lineages = batch_lineages.to(device)
                    b_contrast_lineages = b_contrast_lineages.to(device)
                else:
                    batch_seqs = batch_seqs.to("cuda:0")
                    batch_lineages = batch_lineages.to("cuda:0")
                    b_contrast_lineages = b_contrast_lineages.to("cuda:0")
                b = batch_seqs.size(0)
                weight_f_loss = weight.expand(size=[b, 7]).contiguous().view(-1)
                batch_phy = batch_phy.to(device)
                b_contrast_label = b_contrast_label.to(device)
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    X = torch.cat([batch_seqs, batch_lineages], dim=-1)
                    if gpip is False:
                        (batch_model_output,
                            predict_output_phy,
                            predict_output_contrast,
                         ) = model(X, b_contrast_lineages)  # B, L2-1, C_wordsvocab
                    else:
                        (batch_model_output,
                            predict_output_phy,
                            predict_output_contrast,
                         ) = model(X, b_contrast_lineages).local_value()  # B, L2-1, C_wordsvocab
                    # print(batch_model_output)
                    batch_lineages = batch_lineages.to(device)
                    targets = batch_lineages[:, 1:].contiguous().view(-1)
                    predict_output = batch_model_output.contiguous().view(-1, model_param.a_vocab_size)
                    phy_targets = batch_phy.contiguous().view(-1)
                    contrast_targets = b_contrast_label.contiguous().view(-1)
                    loss1 = loss_func_fce(predict_output, targets, weight_f_loss, "sum") / (b * 6.0)
                    loss2 = loss_func_fce(predict_output_phy, phy_targets)
                    loss3 = loss_func_fce(predict_output_contrast, contrast_targets)
                    loss = loss1 + loss2 + loss3
                    # record losses
                    loss1_val = loss1.item()
                    loss2_val = loss2.item()
                    loss3_val = loss3.item()
                    indicator = loss.clone().detach().isnan().int()
                    if indicator != 0:
                        loss_val = 0.
                    else:
                        loss_val = loss.item()
                    losses += loss_val
                scaler.scale(loss).backward()
                for para in model.parameters():
                    if para.requires_grad is True and para.grad.isnan().int().sum() != 0:
                        para.grad = torch.zeros_like(para.grad).float().to(device)
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                # loss.backward()
                # optimizer.step()
                trainingStep += 1
                acc = accuracy(predict_output, targets)[0].item()
                acc_phy = accuracy(predict_output_phy, phy_targets)[0].item()
                acc_contra = accuracy(predict_output_contrast, contrast_targets)[
                    0
                ].item()
                writer.add_scalar("Loss with training steps", loss_val, trainingStep)
                writer.add_scalar("Accuracy with training steps", acc, trainingStep)
                print(
                    f"[RANK: {local_rank}] | Epoch {e} Iteration({idx}/{n_batches}) "
                    + "| Loss: {:.4f} Loss_g: {:.4f} Loss_p: {:.4f} Loss_c: {:.4f} ".format(
                        losses / trainingStep,
                        loss1_val,
                        loss2_val,
                        loss3_val,
                    )
                    + "| ACC_gen: {:.4f} ACC_phy: {:.4f} ACC_con: {:.4f} | LR lstm: {:.8f} LR GPT {:.8f}".format(
                        acc,
                        acc_phy,
                        acc_contra,
                        optimizer.param_groups[0]["lr"],
                        optimizer.param_groups[1]["lr"],
                    )
                )
                if trainingStep % 10 == 0:
                    with torch.no_grad():
                        p_argmax = torch.argmax(predict_output, dim=-1)[0:7].tolist()
                        t_t = targets[0:7].tolist()

                        phy_argmax = torch.argmax(predict_output_phy, dim=-1)[0:7].tolist()
                        t_p = phy_targets[0:7].tolist()

                        con_argmax = torch.argmax(predict_output_contrast, dim=-1)[0:7].tolist()
                        t_c = contrast_targets[0:7].tolist()
                        print("[RANK: {}] | pre_g: {} truth_g: {}".format(
                            local_rank,
                            p_argmax,
                            t_t,
                        )
                        )
                        print("[RANK: {}] | pre_p: {} truth_p: {}".format(
                            local_rank, phy_argmax, t_p
                        )
                        )
                        print("[RANK: {}] | pre_c: {} truth_c: {}".format(
                            local_rank, con_argmax, t_c
                        )
                        )

        # record losses
        if local_rank in [-1, 0]:
            writer.add_scalar("Loss with epoch", losses / n_batches, e)

        if gpip is False:
            dist.barrier()

        # test part
        assert res_out_path is not None, ValueError(
            f"res_out_path parameter is {res_out_path}."
        )
        ckpt_name = None
        if eval_mode or e % training_param.eval_epoch == 0 or e == training_param.train_epoch:
            if gpip:
                raise ValueError("gpip mode is not suitable for this inference.")
            else:
                if res_out_path[-1] != "/":
                    res_out_path += "/"

                family_out_path = res_out_path + "family__"
                # test for removing clades at family rank
                print(f"Rank {local_rank} start to eval for family.")
                res_out_path_cur = family_out_path + str(local_rank) + ".txt"
                l, a, correct_info = test_TaCoGPT(
                    model.module,
                    training_param,
                    model_param,
                    test_loader_family,
                    e,
                    device,
                    tree_idx,
                    writer,
                    res_out_path_cur,
                    gpip,
                    False,
                )
                dist.barrier()
                if local_rank in [-1, 0]:
                    t_ranks = dist.get_world_size()
                    zero_wh = open(family_out_path + str(0) + ".txt", "a")
                    for r in range(1, t_ranks):
                        res_out_path_cur = family_out_path + str(r) + ".txt"
                        with open(res_out_path_cur, "r") as rh:
                            for line in rh:
                                zero_wh.write(line)
                    zero_wh.close()
                    acc_all = rank_accuracy_calculate_from_test_output_file(
                        family_out_path + str(0) + ".txt", 6
                    )
                    print("Acc All of ranks result at family rank: {}".format(acc_all))
                    ckpt_name = "{}".format(e) + "-f-{:.2f}-{:.2f}-{:.2f}".format(
                        *acc_all[0:3]
                    )
                dist.barrier()

                # test for removing clades at genus rank
                genus_res_path = res_out_path + "genus__"
                print(f"Rank {local_rank} start to eval for genus.")
                res_out_path_cur = genus_res_path + str(local_rank) + ".txt"
                l, a, correct_info = test_TaCoGPT(
                    model.module,
                    training_param,
                    model_param,
                    test_loader_genus,
                    e,
                    device,
                    tree_idx,
                    writer,
                    res_out_path_cur,
                    gpip,
                    False,
                )
                dist.barrier()
                if local_rank in [-1, 0]:
                    t_ranks = dist.get_world_size()
                    zero_wh = open(genus_res_path + str(0) + ".txt", "a")
                    for r in range(1, t_ranks):
                        res_out_path_cur = genus_res_path + str(r) + ".txt"
                        with open(res_out_path_cur, "r") as rh:
                            for line in rh:
                                zero_wh.write(line)
                    zero_wh.close()
                    acc_all = rank_accuracy_calculate_from_test_output_file(
                        genus_res_path + str(0) + ".txt", 6
                    )
                    print("Acc All of ranks result at genus rank: {}".format(acc_all))
                    ckpt_name = ckpt_name + "-g-{:.2f}-{:.2f}-{:.2f}-{:.2f}".format(
                        *acc_all[0:4]
                    )
                dist.barrier()
        # save the model.
        dist.barrier()
        if local_rank in [-1, 0] and eval_mode is False:
            if ckpt_name is None:
                ckpt_name = f"Epoch-{e}"
            torch.save(
                model.module.state_dict(),
                os.path.join(
                    training_param.model_weight_save_folder, ckpt_name
                ),
            )
        if gpip is False:
            dist.barrier()
        if eval_mode:
            break
        else:
            warmUpScheduler.step()
    writer.close()
