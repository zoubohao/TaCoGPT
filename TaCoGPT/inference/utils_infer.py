from copy import deepcopy
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

from TaCoGPT.data.utils_data import convert_indices2seq
from TaCoGPT.IO import readFasta, readVocabulary, writeFasta
from TaCoGPT.model.layers import empty_cache

##################
# inference core #
##################


def decoder_predict(predict_l: torch.Tensor, idx2taxon: Dict[int, str], res: List):
    """_summary_

    Args:
        predict_l (torch.Tensor): [B, 4]
        idx2taxon (Dict[int, str]): _description_
    """
    b, _ = predict_l.shape
    for i in range(b):
        curIndices = predict_l[i].tolist()
        curTokens = []
        for idx in curIndices:
            curTokens.append(idx2taxon[idx])
        res.append(curTokens)


def topk_search(
    model,
    X: torch.Tensor,
    predictions,
    topK: int,
    tree: Dict,
    gpip: bool = False
):
    """
    Implements greedy Search to extend the sequences given in X. The method can compute 
    several outputs in parallel with the first dimension of X.
    Parameters
    ----------    
    X: LongTensor of shape (examples, length)
        The sequences to start the decoding process.
    predictions: int
        The number of tokens to append to X.
    topK: int
        The number of tokens needs to calculate the similarity with seq_rep_norm
    tree: {"TaxoLevel": "superkingdom", "Name": 1, "Children": []} the Name attribute is a int index of the lineage vocabulary.
    Returns
    -------
    X: LongTensor of shape (examples, predictions)
        The sequences extended with the decoding process.
    prob: (examples, predictions)
    """
    device = X.device

    res_idx = []
    res_prob = []

    def inner(cur_X, cur_res_idx, cur_res_prob, tree):
        if len(cur_res_idx) == predictions:
            assert len(cur_res_idx) == len(cur_res_prob)
            res_idx.append(deepcopy(cur_res_idx))
            res_prob.append(deepcopy(cur_res_prob))
            return

        if gpip is False:
            next_probabilities = model(cur_X, None)
        else:
            next_probabilities = model(cur_X, None).local_value()

        # find the topk idx for next generation
        probabilities_generated_topK, generated_topK_idx = next_probabilities.topk(
            k=topK, axis=-1)  # [b, topk], [b, topk]

        candidate_next = set()
        for child in tree["Children"]:
            if isinstance(child, dict):
                candidate_next.add(child["Name"])
            else:
                candidate_next.add(child)

        searching_next_idx = []
        searching_next_prob = []

        generated_topk_idx_list = generated_topK_idx[0, :].tolist()
        generated_topk_prob_list = probabilities_generated_topK[0, :].tolist()

        if generated_topk_idx_list[0] in candidate_next:
            searching_next_idx.append(generated_topk_idx_list[0])
            searching_next_prob.append(generated_topk_prob_list[0])
            max_val = generated_topk_prob_list[0]
            for m in range(1, topK):
                if generated_topk_idx_list[m] in candidate_next \
                        and sum(searching_next_prob) < 0.95 \
                        and abs(max_val - generated_topk_prob_list[m]) <= 0.20:
                    searching_next_idx.append(generated_topk_idx_list[m])
                    searching_next_prob.append(generated_topk_prob_list[m])
        else:
            # make sure the predictions are all valid.
            cur_candidate_probs = []
            for candidate_idx in candidate_next:
                cur_next_probs = next_probabilities.clone()
                if len(cur_next_probs.shape) == 2:
                    cur_next_probs = cur_next_probs.squeeze(0)
                cur_candidate_probs.append((candidate_idx, cur_next_probs[candidate_idx].item()))
            sorted_probs_idx_list = list(
                sorted(cur_candidate_probs, key=lambda x: x[1], reverse=True))

            max_probs_idx = sorted_probs_idx_list[0]
            searching_next_idx.append(max_probs_idx[0])
            searching_next_prob.append(max_probs_idx[1])

        assert len(searching_next_idx) == len(searching_next_prob)

        # searching
        for i, next_idx_int in enumerate(searching_next_idx):
            cur_res_idx.append(next_idx_int)
            cur_res_prob.append(searching_next_prob[i])

            next_token_tensor = torch.tensor(
                next_idx_int, dtype=torch.long, device=device)[None, None]

            child = None
            for child in tree["Children"]:
                if isinstance(child, dict):
                    if child["Name"] == next_idx_int:
                        child = child
                        break
                else:
                    if child == next_idx_int:
                        child = child
                        break

            next_X = torch.cat([cur_X, next_token_tensor], dim=-1)
            inner(next_X, cur_res_idx, cur_res_prob, child)

            cur_res_idx.pop(-1)
            cur_res_prob.pop(-1)

    model.eval()
    assert X.size(0) == 1, "The number of batch size must be 1."
    inner(X, [], [], tree)
    if gpip is False:
        return res_idx, res_prob, model.cache_seq_norm_rep

    mds = [*model][-1]
    return res_idx, res_prob, mds.cache_seq_norm_rep


##############################################
# Beam search for generating the DNA tokens. #
##############################################


def beam_search(
    model,
    X,
    predictions=20,
    beam_width=4,
    verbose=False
):
    """
    Implements Beam Search to extend the sequences given in X. The method can compute 
    several outputs in parallel with the first dimension of X.

    Parameters
    ----------    
    X: LongTensor of shape (examples, length)
        The sequences to start the decoding process.

    predictions: int
        The number of tokens to append to X.

    beam_width: int
        The number of candidates to keep in the search. 

    Returns
    -------
    X: LongTensor of shape (examples, beam_width, length + predictions)
        The sequences extended with the decoding process.

    probabilities: FloatTensor of length examples
        The estimated log-probabilities for the output sequences. They are computed by iteratively adding the 
        probability of the next token at every step.
    """

    model.eval()
    empty_cache(model)
    bs = X.size(0)
    with torch.no_grad():
        # The next command can be a memory bottleneck, but can be controlled with the batch
        # size of the predict method.
        next_probabilities = model.forward(X)[:, -1, :]
        vocabulary_size = next_probabilities.shape[-1]
        probabilities, idx = next_probabilities.squeeze().log_softmax(-1).topk(k=beam_width, axis=-1)
        X = X.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim=-2)
        next_chars = idx.reshape(-1, 1)
        X = torch.cat((X, next_chars), axis=-1)
        # This has to be minus one because we already produced a round
        # of predictions before the for loop.
        predictions_iterator = range(predictions - 1)
        for i in predictions_iterator:
            if verbose and i % 10 == 0:
                print(f"Generated at {i}-th token, total tokens: {predictions}")
            dataset = tud.TensorDataset(X)
            loader = tud.DataLoader(dataset, batch_size=bs)
            next_probabilities = []
            iterator = iter(loader)
            for (x,) in iterator:
                next_probabilities.append(model.forward(x)[:, -1, :].log_softmax(-1))
            next_probabilities = torch.cat(next_probabilities, axis=0)
            next_probabilities = next_probabilities.reshape(
                (-1, beam_width, next_probabilities.shape[-1]))
            probabilities = probabilities.unsqueeze(-1) + next_probabilities
            probabilities = probabilities.flatten(start_dim=1)
            probabilities, idx = probabilities.topk(k=beam_width, axis=-1)
            next_chars = torch.remainder(idx, vocabulary_size).flatten().unsqueeze(-1)
            best_candidates = (idx / vocabulary_size).long()
            best_candidates += torch.arange(X.shape[0] // beam_width,
                                            device=X.device).unsqueeze(-1) * beam_width
            X = X[best_candidates].flatten(end_dim=-2)
            X = torch.cat((X, next_chars), axis=1)
        return X.reshape(-1, beam_width, X.shape[-1]), probabilities


def beam_search_generation(
        seq: str,
        gpt_model: nn.Module = None,
        gpt_kmer_k: int = 4,
        gpt_kmer2index: Dict[str, int] = None,
        beam_width: int = 2,
        prompt_len: int = 800,
        generating_len: int = 100,
        device="cpu",
        verbose=True):

    assert gpt_model is not None, ValueError("GPT model is None.")
    gpt_model = gpt_model.to(device)
    base_gap = 1
    gpt_index2kmer = {}
    for key, val in gpt_kmer2index.items():
        gpt_index2kmer[int(val)] = key

    # generating
    prev_seq = seq[0: -prompt_len]
    sub_seq = seq[-prompt_len:]
    if len(sub_seq) != prompt_len:
        return seq
    else:
        indices = []
        for k in range(0, len(sub_seq) - gpt_kmer_k + 1, base_gap):
            nts = sub_seq[k: k + gpt_kmer_k]
            if nts in gpt_kmer2index:
                indices.append(gpt_kmer2index[nts])
            else:
                indices.append(gpt_kmer2index["<unk>"])
        indices_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

    # generating, generated_seq: [1, beam_width, prompt_len + generating_len]
    generated_seq, _ = beam_search(gpt_model, indices_tensor,
                                   generating_len, beam_width, verbose=verbose)
    generated_seq = generated_seq[:, 0, :]  # get the max log-prob
    if verbose:
        print("Generated Seq Indices: ", generated_seq[0, -100:])
    seqs_string = convert_indices2seq(generated_seq, gpt_index2kmer, 1)[0]
    cur_seq = prev_seq + seqs_string
    return cur_seq


def top_k_top_p_filtering(logits, top_k=4, top_p=0.95, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)

        Basic outline taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 2  # [BATCH_SIZE, VOCAB_SIZE]
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Replace logits to be removed with -inf in the sorted_logits
    sorted_logits[sorted_indices_to_remove] = filter_value
    # Then reverse the sorting process by mapping back sorted_logits to their original position
    logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))

    # print(torch.sort(F.softmax(logits, -1), descending=True))
    pred_token = torch.multinomial(F.softmax(logits, -1), 1)  # [BATCH_SIZE, 1]
    return pred_token


def top_k_p_sampling_generation(
        seq: str,
        gpt_model: nn.Module = None,
        gpt_kmer_k: int = 4,
        gpt_kmer2index: Dict[str, int] = None,
        top_k: int = 4,
        top_p: float = 0.,
        prompt_len: int = 800,
        generating_len: int = 100,
        device="cpu",
        verbose=False):

    assert gpt_model is not None, ValueError("GPT model is None.")
    gpt_model = gpt_model.to(device)
    base_gap = 1
    gpt_index2kmer = {}
    for key, val in gpt_kmer2index.items():
        gpt_index2kmer[int(val)] = key

    # generating
    prev_seq = seq[0: -prompt_len]
    sub_seq = seq[-prompt_len:]
    if len(sub_seq) != prompt_len:
        return seq
    indices = []
    for k in range(0, len(sub_seq) - gpt_kmer_k + 1, base_gap):
        nts = sub_seq[k: k + gpt_kmer_k]
        if nts in gpt_kmer2index:
            indices.append(gpt_kmer2index[nts])
        else:
            indices.append(gpt_kmer2index["<unk>"])
    indices_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(generating_len):
        next_logits = gpt_model(indices_tensor)[:, -1, :]
        pred_token = top_k_top_p_filtering(next_logits, top_k, top_p)
        indices_tensor = torch.cat([indices_tensor, pred_token], dim=-1)

    if verbose:
        print("Generated Seq Indices: ", indices_tensor[0, -100:])
    seqs_string = convert_indices2seq(indices_tensor, gpt_index2kmer, 1)[0]
    cur_seq = prev_seq + seqs_string
    return cur_seq
