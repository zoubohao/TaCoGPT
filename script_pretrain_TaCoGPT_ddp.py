import math
import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from TaCoGPT.data.datasets import SeqSampledPretrainDataset
from TaCoGPT.IO import readVocabulary
from TaCoGPT.model.models import TaCoGPT
from TaCoGPT.train.pretrain import (get_total_params, pretrain_TaCoGPT_test,
                                    pretrain_TaCoGPT)

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"


def main(local_rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"

    class TrainArgs:
        genome_folder_path: str = "../original_final_data/"
        model_weight_save_folder: str = "./InfoFiles_Final/pretrain/"
        model_weight_load_path: str = ""
        kmer_vocab_path: str = "./InfoFiles_Final/kmer_vocab.txt"
        kmer_k: int = 3
        reverse_comp_dict: dict = readVocabulary("./InfoFiles_Final/kmerIdx2revIdx.txt")
        learn_rev_seq: bool = False
        lineage_n: int = 6  # how many taxonomic ranks
        seq_max_len: int = 4096
        train_epoch: int = 40
        train_repeat_time_per_epoch: int = 1
        batch_size: int = 2
        regu_lambda: float = 1e-5
        lr: float = 1e-6
        lr_multiple: float = 1.5
        lr_warmup_epoch: int = 2
        loss_gamma: float = 1.0
        loss_state: str = "mean"  # mean or sum
        eval_epoch: int = 4
        trainOReval: str = "train"
        device = None

    # # 7B {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-06, "vocab_size": -1}

    class ModelArgs:
        dim: int = 2048
        n_layers: int = 18
        n_heads: int = 32
        lineage_n: int = 6
        multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
        norm_eps: float = 1e-6
        dropout_prob: float = 0.125
        pad_id: int = 0
        pretrain_TaCoGPT: bool = True
        cache_infer: bool = False
        # ONLY used if cache_infer is True. This parameter is used to control the gpt DNA generating process but not taxonomic classifications.
        cache_infer_gpt: bool = False
        # auto
        k_vocab_size: int = len(readVocabulary(TrainArgs.kmer_vocab_path))
        model_max_seq_len: int = TrainArgs.seq_max_len // 8
        seq_max_len: int = TrainArgs.seq_max_len

    # those code must run first.
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
    device = torch.device("cuda", local_rank)

    if TrainArgs.device is not None:
        device = TrainArgs.device

    model = TaCoGPT(params=ModelArgs)
    model = model.to(device)
    print("Total parameters: ", get_total_params(model))

    if (
        TrainArgs.model_weight_load_path is not None
        and TrainArgs.model_weight_load_path != ""
        and local_rank in [-1, 0]
    ):
        print("Load weight to the node...")
        state = torch.load(TrainArgs.model_weight_load_path, map_location=device)
        model.load_state_dict(state, strict=False)

    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    if TrainArgs.trainOReval == "train":
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    train_dataset = SeqSampledPretrainDataset(
        TrainArgs.genome_folder_path,
        TrainArgs.seq_max_len,
        TrainArgs.kmer_vocab_path,
        TrainArgs.kmer_k,
        TrainArgs.reverse_comp_dict,
        TrainArgs.learn_rev_seq,
        "train",
    )

    test_dataset = SeqSampledPretrainDataset(
        TrainArgs.genome_folder_path,
        TrainArgs.seq_max_len,
        TrainArgs.kmer_vocab_path,
        TrainArgs.kmer_k,
        TrainArgs.reverse_comp_dict,
        TrainArgs.learn_rev_seq,
        "test",
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    # test_sampler = DistributedSampler_pretrain(test_dataset, shuffle=False)

    train_dist_loader = DataLoader(
        train_dataset,
        TrainArgs.batch_size,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        shuffle=False
    )
    test_loader = DataLoader(test_dataset, TrainArgs.batch_size,
                             num_workers=8, pin_memory=True, shuffle=True)

    if TrainArgs.trainOReval.lower() == "train":
        pretrain_TaCoGPT(model, TrainArgs, ModelArgs, train_dist_loader,
                         test_loader, device, local_rank)
    else:
        if local_rank in [-1, 0]:
            pretrain_TaCoGPT_test(model, TrainArgs, ModelArgs, test_loader, 1, device, None, None)


if __name__ == "__main__":
    world_size = 6
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
