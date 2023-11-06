import os
import random
from copy import deepcopy
from math import floor
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from TaCoGPT.data.utils_data import (convert_seq2ids,
                                     convertLineageToIndexTensor,
                                     randomlyReturnNegTaxoWithDiffPhy,
                                     randomReturnNegTaxoAtSameLevel,
                                     randomReturnNegTaxoWithSamePhy,
                                     sampleSeqFromFasta, seqTokenReplace,
                                     seqValid)
from TaCoGPT.inference.utils_infer import beam_search_generation
from TaCoGPT.IO import readFasta, readPickle, readTXT, readVocabulary


class SeqSampledDataset(Dataset):
    def __init__(
        self,
        fasta_folder_path: str,
        fasta2lineage_path: str,  # csv file
        lineage_vocab_path: str,
        seq_max_len: int,
        kmer_vocab_path: str,
        kmer_k: int,
        tree_pkl_path: str,
        num_nge: int,
        reverse_comp_dict: dict,
        learn_rev_comp_seq=False,
        sampled_times=None,
        train_test="train"
    ) -> None:
        super().__init__()
        self.kmer_k = kmer_k
        self.sampled_times = sampled_times
        self.folder_path = fasta_folder_path
        self.fasta2lineage_path = fasta2lineage_path
        self.seq_max_len = seq_max_len
        self.state = train_test
        self.data_meta_info = []
        self.num_neg = num_nge
        self.learn_rev = learn_rev_comp_seq
        # read info
        self.kmer2index = readVocabulary(kmer_vocab_path)
        self.taxon2index = readVocabulary(lineage_vocab_path)
        self.tree = readPickle(tree_pkl_path)
        self.phyla_set = set()
        for child in self.tree["Children"]:
            self.phyla_set.add(child["Name"])
        self.num_phylum = len(self.phyla_set)

        if train_test == "train":
            self.convert_lineage2indices_train()
        else:
            self.convert_lineage2indices_test()

        self.reverse_comp_dict = {}
        for k, v in reverse_comp_dict.items():
            self.reverse_comp_dict[int(k)] = int(v)

    def convert_lineage2indices_train(self):
        with open(self.fasta2lineage_path, "r") as rh:
            for line in rh:
                info = line.strip("\n").split(",")
                file_name = info[0]
                lineages = info[1:]
                curList = [1]
                for i, l in enumerate(lineages):
                    curList.append(self.taxon2index[l])
                curList.append(2)
                lin_tensor = torch.tensor(data=curList, dtype=torch.long)
                phy_tensor = torch.tensor(data=self.taxon2index[lineages[0]])
                if os.path.exists(os.path.join(self.folder_path, file_name)):
                    self.data_meta_info.append((file_name, lin_tensor, phy_tensor, lineages))

    def convert_lineage2indices_test(self):
        with open(self.fasta2lineage_path, "r") as rh:
            for line in rh:
                info = line.strip("\n").split(",")
                file_name = info[0]
                lineages = info[1:]

                curList = [1]
                for i, l in enumerate(lineages):
                    if l in self.taxon2index:
                        curList.append(self.taxon2index[l])
                    else:
                        curList.append(self.taxon2index["unknown"])
                curList.append(2)
                lin_tensor = torch.tensor(data=curList, dtype=torch.long)
                phy_tensor = torch.tensor(data=self.taxon2index[lineages[0]])
                for i in range(self.sampled_times):
                    prefix, _ = os.path.splitext(file_name)
                    path = os.path.join(self.folder_path, prefix + "." + str(i))
                    # print(path, os.path.exists(path))
                    if os.path.exists(path):
                        self.data_meta_info.append(
                            (prefix + "." + str(i), lin_tensor, phy_tensor, ""))

    def convert_seq2indices(self, seq: str):
        n = len(seq)
        if n > self.seq_max_len - 2:
            n = self.seq_max_len - 2
        k = 1
        indices = [1]
        for i in range(n - self.kmer_k + 1):
            nts = seq[i: i + self.kmer_k]
            if nts in self.kmer2index:
                indices.append(self.kmer2index[nts])
            else:
                indices.append(self.kmer2index["<unk>"])
            k += 1
        indices.append(2)
        k += 1

        reverse_seq = None
        if self.learn_rev:
            reverse_seq = list(
                reversed(list(map(lambda x: self.reverse_comp_dict[x], indices[1:-1]))))
            reverse_seq.insert(0, 1)
            reverse_seq.append(2)
            assert len(indices) == len(reverse_seq)

        if self.seq_max_len > k:
            padd_num = self.seq_max_len - k
            for _ in range(padd_num):
                indices.append(0)
                if self.learn_rev:
                    reverse_seq.append(0)

        assert len(indices) == self.seq_max_len, "len indices {}, seq_max_len {}".format(
            len(indices), self.seq_max_len)
        indices_tensor = torch.tensor(indices, dtype=torch.long)

        reverse_seq_tensor = None
        if self.learn_rev:
            reverse_seq_tensor = torch.tensor(reverse_seq, dtype=torch.long)

        return indices_tensor, reverse_seq_tensor

    def _get_random_num(self, fixed_val: int = None):
        if fixed_val is None:
            return np.random.choice(6, None, replace=False, p=[0.07, 0.08, 0.09, 0.12, 0.13, 0.51]) + 1
        assert fixed_val <= 6 and fixed_val >= 0
        return fixed_val

    def __getitem__(self, index):
        file_name, lineage_tensor, p_tensor, lineages = self.data_meta_info[index]
        if self.state == "train":
            misMatchTensorList = []
            # randomly select truth sub-lineage from lineages
            matchTaxoLevel = self._get_random_num()
            matchLineageList = lineages[0:matchTaxoLevel]

            # randomly select negative lineages from tree at the same taxonomic rank
            sameLevelMisMatches = randomReturnNegTaxoAtSameLevel(
                matchLineageList, self.num_neg // 2, self.tree)
            for misMatchText in sameLevelMisMatches:
                misMatchTensorList.append(
                    convertLineageToIndexTensor(self.taxon2index, misMatchText))

            # cal the vals
            oriPhy = lineages[0]
            copyPhys = deepcopy(self.phyla_set)
            copyPhys.remove(oriPhy)
            mismatchPhylums = list(copyPhys)
            curNum = len(misMatchTensorList)
            left = self.num_neg - curNum
            num_diff_phy = floor(left * 0.35) + 1
            num_sam_phy = self.num_neg - curNum - num_diff_phy

            # randomly select negative lineages from tree at the a random taxonomic rank with different phylum
            for _ in range(num_diff_phy):
                misMatchTaxoLevel = self._get_random_num()
                random.shuffle(mismatchPhylums)
                startPhylum = mismatchPhylums[np.random.randint(self.num_phylum - 1)]
                misMatchText = randomlyReturnNegTaxoWithDiffPhy(
                    self.tree, startPhylum, misMatchTaxoLevel, lineages)
                misMatchTensorList.append(
                    convertLineageToIndexTensor(self.taxon2index, misMatchText))

            # randomly select negative lineages from tree at the a random taxonomic rank with same phylum
            for _ in range(num_sam_phy):
                misMatchTaxoLevel = np.random.choice(
                    5, 1, replace=False, p=[0.1, 0.11, 0.12, 0.17, 0.5]) + 2
                misMatchTaxoLevel = misMatchTaxoLevel[0]
                misMatchText = randomReturnNegTaxoWithSamePhy(
                    self.tree, oriPhy, misMatchTaxoLevel, lineages)
                if misMatchText is not None:
                    misMatchTensorList.append(
                        convertLineageToIndexTensor(self.taxon2index, misMatchText))
                else:
                    random.shuffle(mismatchPhylums)
                    misMatchTaxoLevel = self._get_random_num()
                    startPhylum = mismatchPhylums[np.random.randint(self.num_phylum - 1)]
                    misMatchText = randomlyReturnNegTaxoWithDiffPhy(
                        self.tree, startPhylum, misMatchTaxoLevel, lineages)
                    misMatchTensorList.append(
                        convertLineageToIndexTensor(self.taxon2index, misMatchText))
            textList = misMatchTensorList
            random.shuffle(textList)

            insertIndex = np.random.choice(self.num_neg + 1, 1, replace=False)
            matchTextTensor = convertLineageToIndexTensor(self.taxon2index, matchLineageList)
            textList.insert(insertIndex[0], matchTextTensor)
            contrast_lineages = torch.stack(textList, dim=0)

            # print(textList)
            # print(insertIndex)
            # print(textList[insertIndex[0]], matchTextTensor)
            # print(lineage_tensor)

            contrast_label = torch.tensor(insertIndex[0], dtype=torch.int64)
            # Can add data augmentation at here.
            seq = sampleSeqFromFasta(os.path.join(self.folder_path, file_name),
                                     self.seq_max_len, 1000)
            seq = seq.upper()

            while not seqValid(seq, 0.1):
                seq = sampleSeqFromFasta(os.path.join(self.folder_path, file_name),
                                         self.seq_max_len, 1000)
                seq = seq.upper()

            seq = seqTokenReplace(seq, ratio=0.01)

            seq_tensor, reverse_seq_tensor = self.convert_seq2indices(seq)
            if reverse_seq_tensor is not None and random.random() < 0.5:
                seq_tensor = reverse_seq_tensor

            return seq_tensor, lineage_tensor, p_tensor, contrast_lineages, contrast_label
        else:
            seq = list(readFasta(os.path.join(self.folder_path, file_name)).values())[0]
            seq = seq.upper()
            seq_tensor, _ = self.convert_seq2indices(seq)
            return seq_tensor, lineage_tensor, p_tensor, seq

    def __len__(self):
        return len(self.data_meta_info)


class SeqSampledPretrainDataset(Dataset):
    def __init__(
        self,
        fasta_folder_path: str,
        seq_max_len: int,
        kmer_vocab_path: str,
        kmer_k: int,
        reverse_comp_dict: dict,
        learn_rev_comp_seq=False,
        trainORtest="train",
    ) -> None:
        super().__init__()
        self.kmer_k = kmer_k
        self.folder_path = fasta_folder_path
        self.seq_max_len = seq_max_len
        self.data_meta_info = []
        self.genomeid2path = {}
        self.genomes_id_set = set()
        self.learn_rev = learn_rev_comp_seq
        # read info
        self.kmer2index = readVocabulary(kmer_vocab_path)
        self.reverse_comp_dict = {}
        for k, v in reverse_comp_dict.items():
            self.reverse_comp_dict[int(k)] = int(v)
        if trainORtest == "train":
            self.genome_index(8)
        else:
            self.genome_index(1)

    def genome_index(self, rep):
        files = os.listdir(self.folder_path)
        for i, file in enumerate(files):
            self.genomeid2path[i] = os.path.join(self.folder_path, file)
            for j in range(rep):
                self.data_meta_info.append((os.path.join(self.folder_path, file), i))
        self.genomes_id_set = set(list(self.genomeid2path.keys()))

    def convert_seq2indices(self, seq: str):
        n = len(seq)
        if n > self.seq_max_len - 2:
            n = self.seq_max_len - 2
        k = 1
        indices = [1]
        for i in range(n - self.kmer_k + 1):
            nts = seq[i: i + self.kmer_k]
            if nts in self.kmer2index:
                indices.append(self.kmer2index[nts])
            else:
                indices.append(self.kmer2index["<unk>"])
            k += 1
        indices.append(2)
        k += 1

        reverse_seq = None
        if self.learn_rev:
            reverse_seq = list(
                reversed(list(map(lambda x: self.reverse_comp_dict[x], indices[1:-1]))))
            reverse_seq.insert(0, 1)
            reverse_seq.append(2)
            assert len(indices) == len(reverse_seq)

        if self.seq_max_len > k:
            padd_num = self.seq_max_len - k
            for _ in range(padd_num):
                indices.append(0)
                if self.learn_rev:
                    reverse_seq.append(0)

        assert len(indices) == self.seq_max_len, "len indices {}, seq_max_len {}".format(
            len(indices), self.seq_max_len)
        indices_tensor = torch.tensor(indices, dtype=torch.long)

        reverse_seq_tensor = None
        if self.learn_rev:
            reverse_seq_tensor = torch.tensor(reverse_seq, dtype=torch.long)

        return indices_tensor, reverse_seq_tensor

    def __getitem__(self, index):
        file_path, genome_id = self.data_meta_info[index]
        # Can add data augmentation at here.
        seq1 = sampleSeqFromFasta(file_path, self.seq_max_len, 1000)
        seq1 = seq1.upper()

        while not seqValid(seq1, 0.25):
            seq1 = sampleSeqFromFasta(file_path, self.seq_max_len, 1000)
            seq1 = seq1.upper()

        label = None
        seq2 = None
        if random.random() <= 0.5:
            label = 1
            seq2 = sampleSeqFromFasta(file_path, self.seq_max_len, 1000)
        else:
            label = 0
            cur_set = deepcopy(self.genomes_id_set)
            cur_set.remove(genome_id)
            other_idx = random.choice(list(cur_set))
            seq2 = sampleSeqFromFasta(self.genomeid2path[other_idx], self.seq_max_len, 1000)
        seq2 = seq2.upper()
        seq2 = seqTokenReplace(seq2, ratio=0.1)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        seq1_tensor, reverse_seq1_tensor = self.convert_seq2indices(seq1)
        if reverse_seq1_tensor is not None and random.random() < 0.5:
            seq1_tensor = reverse_seq1_tensor

        seq2_tensor, reverse_seq2_tensor = self.convert_seq2indices(seq2)
        if reverse_seq2_tensor is not None and random.random() < 0.5:
            seq2_tensor = reverse_seq2_tensor

        return seq1_tensor, seq2_tensor, label_tensor

    def __len__(self):
        return len(self.data_meta_info)


class GenerationDataset(Dataset):

    def __init__(self,
                 file_folder_path: str,
                 seq_max_len: int,
                 model_len: int,
                 kmer_vocab_path: str,
                 kmer_k: int,
                 base_gap: int,
                 trainORtest="train",) -> None:
        super().__init__()
        self.kmer_k = kmer_k
        self.kmer2index = readVocabulary(kmer_vocab_path)
        self.model_len = model_len
        self.seq_max_len = seq_max_len
        self.folder_path = file_folder_path
        self.base_gap = base_gap
        self.state = trainORtest
        assert self.seq_max_len % self.kmer_k <= self.model_len
        self.data = []
        files = os.listdir(file_folder_path)
        for file in files:
            self.data.append(os.path.join(file_folder_path, file))

    def __getitem__(self, index):
        file_path = self.data[index]
        seq = readTXT(file_path)[0]
        if self.state == "train":
            n = len(seq)
            if n // self.base_gap > self.model_len:
                l = self.seq_max_len
                s = random.randint(0, n - l)
                seq = seq[s: s + l]
            indices = convert_seq2ids(seq, self.kmer_k, self.kmer2index,
                                      self.model_len, self.base_gap)
            return indices
        else:
            seq = seq[0: self.seq_max_len]
            indices = convert_seq2ids(seq, self.kmer_k, self.kmer2index,
                                      self.model_len, self.base_gap)
            return indices

    def __len__(self):
        return len(self.data)
