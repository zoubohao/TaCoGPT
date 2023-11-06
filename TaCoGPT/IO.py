
import pickle
from typing import Dict, List, Tuple, Union


def readFasta(path: str) -> Dict[str, str]:
    """This function is used to read fasta file and
    it will return a dict, which key is the name of seq and the value is the sequence.
    This function would not read the plasmid sequences.
    Args:
        path (str): The path of fasta file.

    Returns:
        Dict[str, str]: _description_
    """
    contig2Seq = {}
    curContig = ""
    curSeq = ""
    with open(path, mode="r") as rh:
        for line in rh:
            curLine = line.strip("\n")
            if ">" == curLine[0]:
                if "plasmid" not in curContig.lower():
                    contig2Seq[curContig] = curSeq
                    curContig = curLine
                curSeq = ""
            else:
                curSeq += curLine
    if "plasmid" not in curContig.lower():
        contig2Seq[curContig] = curSeq
    contig2Seq.pop("")
    return contig2Seq


def writeFasta(name2seq: Dict, writePath: str):
    with open(writePath, "w") as wh:
        for key, val in name2seq.items():
            if key[0] != ">":
                wh.write(">" + key + "\n")
            else:
                wh.write(key + "\n")
            wh.write(val + "\n")


def readVocabulary(path: str) -> Dict:
    vocabulary = {}
    with open(path, mode="r") as rh:
        for line in rh:
            oneLine = line.strip("\n").split("\t")
            vocabulary[oneLine[0]] = int(oneLine[1])
    return vocabulary


def readTable(path: str) -> Dict:
    """
    Equal with readVocabulary
    Args:
        path (str): _description_

    Returns:
        Dict: _description_
    """
    vocabulary = {}
    with open(path, mode="r") as rh:
        for line in rh:
            oneLine = line.strip("\n").split("\t")
            vocabulary[oneLine[0]] = oneLine[1]
    return vocabulary


def readCSV(path: str) -> List[Tuple[str]]:
    out = []
    with open(path, "r") as rh:
        for line in rh:
            info = tuple(line.strip("\n").split(","))
            out.append(info)
    return out


def readTXT(path: str) -> List[str]:
    out = []
    with open(path, "r") as rh:
        for line in rh:
            out.append(line.strip("\n"))
    return out


def loadTaxonomyTree(pkl_path: str) -> Dict:
    rb = open(pkl_path, mode="rb")
    tree = pickle.load(rb)
    rb.close()
    return tree


def readPickle(readPath: str) -> object:
    rh = open(readPath, "rb")
    obj = pickle.load(rh)
    rh.close()
    return obj


def writePickle(writePath: str, obj: object) -> None:
    wh = open(writePath, "wb")
    pickle.dump(obj, wh, pickle.HIGHEST_PROTOCOL)
    wh.flush()
    wh.close()


def writeCSV(file_path: str, data: List[List[object]]):
    with open(file_path, "w") as wh:
        for oneline in data:
            wh.write(",".join(oneline) + "\n")
            