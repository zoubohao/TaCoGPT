
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from TaCoGPT.data.datasets import SeqSampledDataset
from TaCoGPT.IO import readPickle, readVocabulary
from TaCoGPT.model.models import TaCoGPT, GPT
from TaCoGPT.train.finetune import (get_total_params, requires_grad_change,
                                    test_TaCoGPT, train_TaCoGPT)
from TaCoGPT.utils_func import rank_accuracy_calculate_from_test_output_file

os.environ["CUDA_VISIBLE_DEVICES"] = "1,4,5,6,7"


def main(local_rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"

    # For TaCoGPT model
    class TrainArgs:
        fasta2lineage_file_path_train: str = "./InfoFiles_AllClade/filename2lineages_final_resample.csv"  # meta information
        fasta2lineage_file_path_test: str = "./InfoFiles_AllClade/filename2lineages_final.csv"  # meta information
        train_data_folder: str = "../TaCoGPT_Data/MAG_simulation_15780_train/"
        test_data_folder: str = "../TaCoGPT_Data/MAG_simulation_15780_test/"
        test_sample_time_per_genome: int = 3
        model_weight_save_folder: str = "./InfoFiles_AllClade/ckpt/"
        model_weight_load_path: str = ""
        kmer_vocab_path: str = "./InfoFiles_AllClade/kmer_vocab.txt"
        kmer_k: int = 3
        reverse_comp_dict: dict = readVocabulary("./InfoFiles_AllClade/kmerIdx2revIdx.txt")
        learn_rev_seq: bool = False
        lineage_vocab_path: str = "./InfoFiles_AllClade/lineage_vocab.txt"
        lineage_n: int = 6  # how many taxonomic ranks
        tree_pkl_path: str = "./InfoFiles_AllClade/taxonTree.pkl"
        tree_idx_path: str = "./InfoFiles_AllClade/taxonTreeIdxs.pkl"
        contrast_neg_num: int = 149
        seq_max_len: int = 4096
        train_epoch: int = 100
        train_repeat_time_per_epoch: int = 16
        batch_size: int = 4
        regu_lambda: float = 5e-4
        lr: float = 5e-6 
        lr_multiple: float = 2.2 
        lr_warmup_epoch: int = 2 
        loss_state: str = "mean"  # mean or sum
        loss_gamma: float = 0.45 
        search_algo: str = "topk"
        topk: int = 3
        eval_epoch: int = 5
        trainOReval: str = "train"

    # 15780  {"dim":2048, "multiple_of": 256, "n_heads": 32, "n_layers": 21, "norm_eps": 1e-06, "vocab_size": -1}
    # For TaCoGPT model
    class ModelArgs:
        dim: int = 2048
        n_layers: int = 21
        n_heads: int = 32  # 16
        lineage_n: int = 6
        multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
        norm_eps: float = 1e-6
        dropout_prob: float = 0.1  # 0.12, 0.135, 0.13, 0.125
        pad_id: int = 0
        pretrain_TaCoGPT: bool = False  # this parameter is used to control the pretrain task of TaCoGPT but not GPT
        # This parameter is used to control the TaCoGPT to do taxonomic classifications with cache mode.
        cache_infer: bool = True
        # ONLY used if cache_infer is True. This parameter is used to control the gpt DNA generating process but not taxonomic classifications.
        cache_infer_gpt: bool = False
        # auto
        k_vocab_size: int = len(readVocabulary(TrainArgs.kmer_vocab_path))
        a_vocab_size: int = len(readVocabulary(TrainArgs.lineage_vocab_path))
        model_max_seq_len: int = TrainArgs.seq_max_len // 8
        seq_max_len: int = TrainArgs.seq_max_len

    # those code must run first.
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
    device = torch.device("cuda", local_rank)

    model = TaCoGPT(params=ModelArgs)
    model = model.to(device)
    print("Total parameters: ", get_total_params(model))

    if TrainArgs.model_weight_load_path is not None and TrainArgs.model_weight_load_path != "":
        print(f"{local_rank} Load TaCoGPT weights to each node...")
        state = torch.load(TrainArgs.model_weight_load_path, map_location=device)
        model.load_state_dict(state, strict=False)

    model = DistributedDataParallel(model).to(device)

    train_dataset = SeqSampledDataset(
        TrainArgs.train_data_folder,
        TrainArgs.fasta2lineage_file_path_train,
        TrainArgs.lineage_vocab_path,
        TrainArgs.seq_max_len,
        TrainArgs.kmer_vocab_path,
        TrainArgs.kmer_k,
        TrainArgs.tree_pkl_path,
        TrainArgs.contrast_neg_num,
        TrainArgs.reverse_comp_dict,
        TrainArgs.learn_rev_seq,
        None,
        "train"
    )
    test_dataset = SeqSampledDataset(
        TrainArgs.test_data_folder,
        TrainArgs.fasta2lineage_file_path_test,
        TrainArgs.lineage_vocab_path,
        TrainArgs.seq_max_len,
        TrainArgs.kmer_vocab_path,
        TrainArgs.kmer_k,
        TrainArgs.tree_pkl_path,
        TrainArgs.contrast_neg_num,
        TrainArgs.reverse_comp_dict,
        False,
        TrainArgs.test_sample_time_per_genome,
        "test",
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dist_loader = DataLoader(
        train_dataset, TrainArgs.batch_size, sampler=train_sampler, pin_memory=True
    )

    test_sampler = DistributedSampler(test_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, TrainArgs.batch_size,
                             sampler=test_sampler, pin_memory=True)

    if TrainArgs.trainOReval.lower() == "train":
        train_TaCoGPT(model, TrainArgs, ModelArgs, train_dist_loader,
                      test_loader, device, local_rank, False, "./Result_Collect/")
    else:
        train_TaCoGPT(model, TrainArgs, ModelArgs, train_dist_loader,
                      test_loader, device, local_rank, False, "./Result_Collect/", True)


if __name__ == "__main__":
    world_size = 5
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
