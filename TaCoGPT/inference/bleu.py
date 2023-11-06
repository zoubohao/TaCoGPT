import math
import operator
from functools import reduce
from typing import List


def count_ngram(candidate, references, n):
    clipped_count = 0
    count = 0
    r = 0
    c = 0
    for cand_sentence, ref_sentence in zip(candidate, references):
        # Calculate precision for each sentence
        ref_counts = []
        ref_lengths = []
        # Build dictionary of ngram counts
        ngram_d = {}
        ref_lengths.append(len(ref_sentence))
        limits = len(ref_sentence) - n + 1
        # loop through the sentance consider the ngram length
        for i in range(limits):
            ngram = ' '.join(ref_sentence[i:i+n]).lower()
            if ngram in ngram_d.keys():
                ngram_d[ngram] += 1
            else:
                ngram_d[ngram] = 1
        ref_counts.append(ngram_d)
        # candidate
        cand_dict = {}
        limits = len(cand_sentence) - n + 1
        for i in range(0, limits):
            ngram = ' '.join(cand_sentence[i:i + n]).lower()
            if ngram in cand_dict:
                cand_dict[ngram] += 1
            else:
                cand_dict[ngram] = 1
        clipped_count += clip_count(cand_dict, ref_counts)
        count += limits
        r += best_length_match(ref_lengths, len(cand_sentence))
        c += len(cand_sentence)
    if clipped_count == 0:
        pr = 0
    else:
        pr = float(clipped_count) / count
    bp = brevity_penalty(c, r)
    return pr, bp


def clip_count(cand_d, ref_ds):
    """Count the clip count for each ngram considering all references"""
    count = 0
    for m in cand_d.keys():
        m_w = cand_d[m]
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count


def best_length_match(ref_l, cand_l):
    """Find the closest length of reference to that of candidate"""
    least_diff = abs(cand_l-ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l-ref) < least_diff:
            least_diff = abs(cand_l-ref)
            best = ref
    return best


def brevity_penalty(c, r):
    if c > r:
        bp = 1
    else:
        bp = math.exp(1-(float(r)/c))
    return bp


def geometric_mean(precisions):
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))


def BLEU(candidate: List[List[str]], references: List[List[str]], n_gram: int = 3):
    """
    Calculate BLEU score.
    Args:
        candidate (List[List[str]]): the generate candidates. one element is one str list for a sentence.
        references (List[List[str]]): the references. one element is one of the truth answer of a candidate.
        n_gram (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_
    """
    assert len(candidate) == len(references), ValueError(
        "the length of candidate must equal with the length of reference.")
    precisions = []
    for i in range(n_gram):
        pr, bp = count_ngram(candidate, references, i + 1)
        precisions.append(pr)
    bleu = geometric_mean(precisions) * bp
    return bleu


# main execution
if __name__ == '__main__':
    a = [["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]]
    b = [["1", "2", "3", "8", "9", "10", "11", "12"]]
    print(BLEU(a, b))
