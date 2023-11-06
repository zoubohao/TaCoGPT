import os
import random
from copy import deepcopy
from subprocess import Popen
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
from numpy import array
from numpy.random import choice, shuffle

from TaCoGPT.IO import readCSV, readFasta


def splitListEqually(input_list: List, num_parts: int) -> List[List[object]]:
    n = len(input_list)
    step = n // num_parts + 1
    out_list = []
    for i in range(num_parts):
        curList = input_list[i * step: (i + 1) * step]
        if curList:
            out_list.append(curList)
    return out_list


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sampleSeqFromFasta(fasta_path: str, seq_max_len: int, seq_min_len: int):
    """
    Args:
        fasta_path (str): _description_
        seq_max_len (int): _description_
        seq_min_len (int): _description_

    Returns:
        _type_: _description_
    """
    contig2seq = readFasta(fasta_path)
    contigs_list = list(contig2seq.values())
    shuffle(contigs_list)

    contigs_num = len(contigs_list)
    if contigs_num == 1:
        seq = contigs_list[0]
    else:
        length = []
        for contig in contigs_list:
            length.append(len(contig))
        p = array(length, dtype=np.float32) / sum(length)
        l = len(p) * 0.25
        if l < 1:
            l = 1
        elif l > 16:
            l = 16
        p = softmax(p * l)
        index = choice(contigs_num, None, p=p)
        seq = contigs_list[index]

    n = len(seq)
    l = random.randint(seq_min_len, seq_max_len)
    if (n - l) > 0:
        s = random.randint(0, n - l)
    else:
        s = 0
    return seq[s: s + l]


def convert_seq2ids(seq: str, kmer_k: int, kmer2index: Dict[str, int], model_len: int, base_gap: int = 1):
    """
    auto padding zero at the end of seq if the length of it less than model_len.
    Args:
        seq (str): upper string 
    """
    n = len(seq)
    k = n % base_gap
    if k != 0:
        seq = seq[0: -k]
    n -= k
    indices = []
    for i in range(0, n - kmer_k + 1, base_gap):
        nts = seq[i: i + kmer_k]
        if nts in kmer2index:
            indices.append(kmer2index[nts])
        else:
            indices.append(kmer2index["<unk>"])

    k = len(indices)
    if model_len > k:
        padd_num = model_len - k
        for _ in range(padd_num):
            indices.append(0)
    indices_tensor = torch.tensor(indices, dtype=torch.long)
    assert len(indices) <= model_len, ValueError(
        "The number of elements in output longer than model_len. len of indices: {}, model_len: {}".format(len(indices), model_len))
    return indices_tensor


def convert_indices2seq(generated_indices: torch.Tensor, index2kmer: Dict[int, str], base_gap=1):
    """
    Args:
        generated_indices (torch.Tensor): [b, l]
        index2kmer (Dict[int, str]): 
    """
    out_seq = []
    for j, indices in enumerate(generated_indices):
        cur_seq = []
        for i, ind in enumerate(indices.tolist()):
            if i == 0:
                if ind <= 1:
                    cur_seq.append("N" * len(index2kmer[2]))
                else:
                    cur_seq.append(index2kmer[ind])
            else:
                if ind <= 1:
                    cur_seq.append("N")
                else:
                    cur_seq.append(index2kmer[ind][-base_gap:])
        out_seq.append("".join(cur_seq))
    return out_seq


def seqValid(seq: str, ratio=0.25):
    n = len(seq) + 0.
    r = seq.count("N")
    if r / n < ratio:
        return True
    return False


index2Taxo = {1: "phylum", 2: "class", 3: "order", 4: "family", 5: "genus", 6: "species"}


def insert(taxoList: List[str], curDict: Dict) -> None:
    index2TaxoR = {6: "phylum", 5: "class", 4: "order", 3: "family", 2: "genus", 1: "species"}
    length = len(taxoList)
    if length == 1:
        if taxoList[0] not in curDict["Children"]:
            curDict["Children"].append(taxoList[0])
    else:
        signal = True
        for child in curDict["Children"]:
            if child["Name"] == taxoList[0]:
                copyTaxo = deepcopy(taxoList)
                copyTaxo.pop(0)
                insert(copyTaxo, child)
                signal = False
        if signal:
            newDict = {"TaxoLevel": index2TaxoR[length], "Name": taxoList[0], "Children": []}
            copyTaxo = deepcopy(taxoList)
            copyTaxo.pop(0)
            insert(copyTaxo, newDict)
            curDict["Children"].append(newDict)


def taxonomyTreeBuild(split_func: Callable, file_path=None) -> Dict:
    """
    This function is used for buliding a taxonomy tree with the map data structure. Like the json structure.
    The biggest taxonomy level is the superkingdom of bacteria.
    There are 6 sub-level, its are phylum, class, order, family, genus, and species.

    1. For the levels of phylum, class, order, family, and genus, each objects in those level will be represented
    as a map with following attributes:
    "TaxoLevel" -> Depicts the taxonomy level of this object,
    "Name" -> The  name of the object,
    "Children" -> The list of next level objects.
    2. For the species level, since there is no next level for species, therefore, the objects in "Children" attribute of genus
    are just strings, which are the name of the species that belong to corresponding genus.

    split_func: The split function must return a tuple that contains the name of "phylum, class, order, family, genus, and species" in
    this order.
    file_path: the path of taxonomy txt file. Each line must contain the taxonomy of one species. Each line will be pharsed by split_func.
    """
    taxonomyTree = {"TaxoLevel": "superkingdom", "Name": "bacteria", "Children": []}
    with open(file_path, mode="r") as rh:
        k = 0
        for line in rh:
            oneLine = line.strip("\n")
            insert(split_func(oneLine), taxonomyTree)
        k += 1
    return taxonomyTree


def split_file_function(oneLine: str) -> List:
    levelsInfor = oneLine.split(",")
    return levelsInfor


def convertLineageToIndexTensor(vocabulary: Dict, labelText: List) -> torch.Tensor:
    labelTextLength = len(labelText)
    if labelTextLength > 6 or labelTextLength < 0:
        raise ValueError(
            "The length of label text must smaller or equal with 6, since there are only 6 taxonomy level."
        )
    seq = []
    for word in labelText:
        if word in vocabulary:
            seq.append(vocabulary[word])
        else:
            raise ValueError("Word does not in the vocabulary.")
    if labelTextLength < 6:
        seq += [0 for _ in range(6 - labelTextLength)]
    seq = torch.from_numpy(np.array(seq, dtype=np.int64))
    return seq


def randomReturnNegTaxoAtSameLevel(matchTextOuter: List, maxNum: int, taxoTree: Dict):
    """
    randomly return the taxonomic lineages at the same rank with matchText
    Args:
        matchTextOuter (List): _description_
        maxNum (int): _description_
        taxoTree (Dict): _description_

    Returns:
        _type_: _description_
    """
    results = []

    def inner(matchTextInner, taxoTree):
        children = taxoTree["Children"]
        if len(matchTextInner) == 1:
            for child in children:
                if isinstance(child, Dict):
                    if child["Name"] != matchTextInner[-1]:
                        results.append(matchTextOuter[0:-1] + [child["Name"]])
                else:
                    if child != matchTextInner[-1]:
                        results.append(matchTextOuter[0:-1] + [child])
        else:
            for child in children:
                if child["Name"] == matchTextInner[0]:
                    inner(matchTextInner[1:], child)

    inner(matchTextOuter, taxoTree)
    random.shuffle(results)
    return results[0:maxNum]


def randomReturnNegTaxoWithSamePhy(
    taxoTree: Dict, startPhylum: str, stopLevel: int, truthInfo: List
) -> Union[List[str], None]:
    assert startPhylum == truthInfo[0], ValueError("Must with same phylum name.")
    res = []
    phys = taxoTree["Children"]
    signal = True
    startPhyTree = None
    for child in phys:
        if child["Name"] == startPhylum:
            startPhyTree = child
            signal = False
    if signal:
        raise ValueError("This phylum name is not in taxonomy tree.")
    # This means you must select other phylum as neg sample since the current match text is just at phy level.
    if stopLevel <= 1 or stopLevel > 6:
        raise ValueError(
            "stop level error. stop level: {}, but needs 1 < stop level <= 6".format(stopLevel))

    def inner(curTaxoTree):
        if isinstance(curTaxoTree["Children"][0], Dict):
            nextLevel = curTaxoTree["Children"][0]["TaxoLevel"]
        else:
            nextLevel = "species"
        # print(nextLevel, index2Taxo[stopLevel])
        if nextLevel != index2Taxo[stopLevel]:
            curChildren = curTaxoTree["Children"]
            nextIndex = np.random.randint(len(curChildren))
            res.append(curTaxoTree["Name"])
            inner(curChildren[nextIndex])
        else:
            curChildren = curTaxoTree["Children"]
            newChildren = []
            for child in curChildren:
                name = child
                if isinstance(child, Dict):
                    name = child["Name"]
                if name != truthInfo[stopLevel - 1]:
                    newChildren.append(child)
            if len(newChildren) == 0:
                return res.append(None)
            nextIndex = np.random.randint(len(newChildren))
            res.append(curTaxoTree["Name"])
            if isinstance(newChildren[nextIndex], str):
                res.append(newChildren[nextIndex])
            else:
                res.append(newChildren[nextIndex]["Name"])

    inner(startPhyTree)
    if res[-1] is None:
        return None
    else:
        return res


def randomlyReturnNegTaxoWithDiffPhy(taxoTree: Dict, startPhylum: str, stopLevel: int, truthInfo: List) -> List[str]:
    """
    randomly return the negative taxonomic lineages with different phylum of the start phylum.
    Args:
        taxoTree (Dict): _description_
        startPhylum (str): _description_
        stopLevel (int): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        List[str]: _description_
    """
    assert startPhylum != truthInfo[0], ValueError("Must with different phylum name.")
    res = []
    phys = taxoTree["Children"]
    signal = True
    startPhyObj = None
    for child in phys:
        if child["Name"] == startPhylum:
            startPhyObj = child
            signal = False
    if signal:
        raise ValueError("This phylum name is not in taxonomy tree.")
    if stopLevel < 1 or stopLevel > 6:
        raise ValueError("stop level error.")

    def inner(curTaxoTree):
        if isinstance(curTaxoTree, Dict):
            curLevel = curTaxoTree["TaxoLevel"]
            curChildren = curTaxoTree["Children"]
            nextIndex = np.random.randint(len(curChildren))
            res.append(curTaxoTree["Name"])
            if curLevel != index2Taxo[stopLevel]:
                inner(curChildren[nextIndex])
        else:
            res.append(curTaxoTree)

    inner(startPhyObj)
    return res


def oversample_phylum(file_path: str, output_path: str, ratio: float, phy_idx=1):
    """_summary_

    Args:
        file_path (str): _description_
        output_path (str): _description_
        ratio (float): random oversample the species in a phylum until the ratio between the number of species divide the max number reach this value.
        phy_idx (int, optional): the index of phylum in taxonomic rank. Defaults to 1.
    """
    metaInfo = readCSV(file_path)
    phy2info = {}
    for info in metaInfo:
        phy = info[phy_idx]
        if phy not in phy2info:
            phy2info[phy] = [info]
        else:
            phy2info[phy].append(info)

    len_info = []
    for key, val in phy2info.items():
        len_info.append((key, len(val)))
    len_info = list(sorted(len_info, key=lambda x: x[-1]))
    max_n = len_info[-1][-1]

    print("phylum max instances num: ", max_n)

    for key, val in phy2info.items():
        cur_n = len(val)
        o = len(val)
        cur_val = deepcopy(val)

        if (cur_n / max_n + 0.0) < ratio:
            while (cur_n / max_n + 0.0) < ratio:
                index = random.randint(0, o - 1)
                cur_val.append(val[index])
                cur_n += 1
            phy2info[key] = cur_val

    with open(output_path, "w") as wh:
        for key, vals in phy2info.items():
            for val in vals:
                n = len(val) - 1
                for i, ele in enumerate(val):
                    if i != n:
                        wh.write(ele + ",")
                    else:
                        wh.write(ele)
                wh.write("\n")


nt2ntList = {"A": ["T", "C", "G"], "T": ["A", "C", "G"], "C": ["T", "A", "G"], "G": ["T", "C", "A"]}
nt = ["T", "C", "G", "A"]


def seqTokenReplace(seq: str, ratio=0.05) -> str:
    def inner(c):
        if random.random() >= ratio:
            return c
        else:
            index = np.random.randint(0, 3, dtype=np.int64)
            if c in nt2ntList:
                return nt2ntList[c][index]
            else:
                return c

    return "".join(map(inner, seq))


def maskSeq(seq: str, ratio=0.05):
    def inner(c):
        if random.random() >= ratio:
            return c
        else:
            return "E"
    return "".join(map(inner, seq))
