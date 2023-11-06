import os
from subprocess import Popen
from typing import List

import numpy as np
import sklearn.metrics as metrics
from TaCoGPT.IO import readFasta, readVocabulary


def getTaxoID2lineage(ids: List[int], output_path: str):
    with open(output_path, "w") as wh:
        for ele in ids:
            wh.write(str(ele) + "\n")
    res = Popen(
        'taxonkit lineage {} | taxonkit reformat -f "{{k}}\t{{p}}\t{{c}}\t{{o}}\t{{f}}\t{{g}}\t{{s}}" -F -P > {}'.format(
            output_path, "./lineage.txt"
        ),
        shell=True,
    )
    res.wait()
    res.kill()
    id2lineage = {}
    with open("./lineage.txt", "r", encoding="utf-8") as rh:
        for line in rh:
            info = line.strip("\n").split("\t")
            if info[0] != "0":
                cur = [ele.replace(" ", "-") for ele in info[2:]]
                l = ";".join(cur)
            else:
                l = ";".join(
                    ["k__unknown", "p__unknown", "c__unknown", "o__unknown",
                        "f__unknown", "g__unknown", "s__unknown"]
                )
            id2lineage[info[0]] = l
    with open(output_path, "w") as wh:
        for key, val in id2lineage.items():
            wh.write(key + "\t" + val + "\n")
    return id2lineage


"""
GPTyrex test accuracy in each rank.
rank_accuracy_calculate_from_test_output_file("./gptyrex_test_output.txt")
"""


def rank_accuracy_calculate_from_test_output_file(output_file: str, num_rank=6):
    correct_info = [0. for _ in range(num_rank)]
    total_info = [0. for _ in range(num_rank)]
    with open(output_file, "r") as rh:
        for line in rh:
            pred, truth = line.strip("\n").split("\t")
            pred = pred.split(",")
            truth = truth.split(",")
            for i, (p, t) in enumerate(zip(pred, truth)):
                if p == t:
                    correct_info[i] += 1
                total_info[i] += 1

    c = np.array(correct_info, dtype=np.float32)
    t = np.array(total_info, dtype=np.float32)
    r = c / t
    print("ACC : p__{:4f}, c__{:4f}, o__{:4f}, f__{:4f}, g__{:4f}, s__{:4f}".format(*r))
    return r


"""
# kraken2 data format
# >sequence16|kraken:taxid|32630  Adapter sequence
# CAAGCAGAAGACGGCATACGAGATCTTCGAGTGACTGGAGTTCCTTGGCACCCGAGAATTCCA...

# kraken2 commands (create custom database.)
for file in chr*.fa
do
    kraken2-build --add-to-library $file --db test_kraken_db
done


kraken2-build --download-taxonomy --db test_kraken_db
kraken2-build --build --db test_kraken_db

kraken2 -db ./kraken2_database/test_kraken_db/ --threads 32 --output ./test_output.txt ./test.fa
kraken2_calculate_accuracy("./test_output.txt") ## calculate the accuracy


kraken2 -db ./kraken2_database/test_kraken_db/ --threads 32 --classified-out ./test_classified.txt --output ./test_output.txt --report test_report.txt -use-mpa-style --use-name ./test.fa
"""

# This function can transform data into a fasta file.
# All tools need to use this function to create database.


def db_data_format_transform(input_folder: str, output_fasta_path: str):
    file_list = os.listdir(input_folder)
    n = len(file_list)
    i = 0
    with open(output_fasta_path, "w") as wh:
        for k, file in enumerate(file_list):
            print(f"({k})/({n}), {file}")
            name2seq = readFasta(os.path.join(input_folder, file))
            taxo_id = file.split(".")[0]
            for key, seq in name2seq.items():
                out_name = f">seq_{str(i)}|kraken:taxid|{taxo_id}"
                wh.write(out_name + "\n")
                wh.write(seq + "\n")
                i += 1


def kraken2_calculate_accuracy(kraken2_output_txt_path: str, 
                               test_fasta_path: str,  output_path: str,
                               num_rank=6):
    truth = []
    predict = []
    vocab2index = {}
    output_set = set()
    with open(kraken2_output_txt_path, "r") as rh:
        for line in rh:
            info = line.strip("\n").split("\t")
            # print(info)
            output_set.add(info[1])
            truth.append(int(info[1].split("|")[-1]))
            predict.append(int(info[2]))

    ids_set = set(list(readFasta(test_fasta_path).keys()))
    for id_ele in ids_set:
        if id_ele[1:] not in output_set:
            truth.append(int(id_ele.split("|")[-1]))
            predict.append(0)

    id2lineage = getTaxoID2lineage(truth + predict, "./kraken2_id2lineage.txt")
    correct_info = [0. for _ in range(num_rank)]
    total_info = [0. for _ in range(num_rank)]
    
    predict_lineages = []
    truth_lineages = []
    k = 0
    
    for p, t in zip(predict, truth):
        # print(p, t)
        p_line = id2lineage[str(p)].split(";")[1:]  # since 0 index is kingdom, start from phylum
        t_line = id2lineage[str(t)].split(";")[1:]
        
        for ele in p_line:
            if ele not in vocab2index:
                vocab2index[ele] = k
                k += 1
        
        for ele in t_line:
            if ele not in vocab2index:
                vocab2index[ele] = k
                k += 1
        
        cur_p_index = []
        cur_t_index = []
        
        for j, (p_l, t_l) in enumerate(zip(p_line, t_line)):
            cur_p_index.append(vocab2index[p_l])
            cur_t_index.append(vocab2index[t_l])
            if p_l == t_l:
                correct_info[j] += 1
            total_info[j] += 1
        
        predict_lineages.append(cur_p_index)
        truth_lineages.append(cur_t_index)
        
    c = np.array(correct_info, dtype=np.float32)
    t = np.array(total_info, dtype=np.float32)
    r = c / t
    
    with open(output_path, "w") as wh:
        for p_lineage, t_lineage in zip(predict_lineages, truth_lineages):
            p_str = ",".join(list(map(lambda x: str(x), p_lineage)))
            t_str = ",".join(list(map(lambda x: str(x), t_lineage)))
            wh.write(p_str + "\t" + t_str + "\n")
    
    print("ACC : p__{:4f}, c__{:4f}, o__{:4f}, f__{:4f}, g__{:4f}, s__{:4f}".format(*r))
    return r

"""
minimap2 db.fasta test.fasta > approx-mapping.paf
"""


def minimap2_calculate_accuracy(minimap2_paf_path: str, test_fasta_path, output_path: str, num_rank=6):
    truth = []
    predict = []
    output_set = set()
    vocab2index = {}
    
    with open(minimap2_paf_path, "r") as rh:
        for line in rh:
            info = line.strip("\n").split("\t")
            t = info[0]
            if t not in output_set:
                output_set.add(t)
                truth.append(int(t.split("|")[-1]))
                predict.append(int(info[5].split("|")[-1]))

    ids_set = set(list(readFasta(test_fasta_path).keys()))
    for id_ele in ids_set:
        if id_ele[1:] not in output_set:
            truth.append(int(id_ele.split("|")[-1]))
            predict.append(0)
    id2lineage = getTaxoID2lineage(truth + predict, "./minimap2_id2lineage.txt")

    correct_info = [0. for _ in range(num_rank)]
    total_info = [0. for _ in range(num_rank)]
    predict_lineages = []
    truth_lineages = []
    k = 0
    for p, t in zip(predict, truth):
        # print(p, t)
        p_line = id2lineage[str(p)].split(";")[1:]  # since 0 index is kingdom, start from phylum
        t_line = id2lineage[str(t)].split(";")[1:]
        for ele in p_line:
            if ele not in vocab2index:
                vocab2index[ele] = k
                k += 1
        
        for ele in t_line:
            if ele not in vocab2index:
                vocab2index[ele] = k
                k += 1
        cur_p_index = []
        cur_t_index = []
        for j, (p_l, t_l) in enumerate(zip(p_line, t_line)):
            
            cur_p_index.append(vocab2index[p_l])
            cur_t_index.append(vocab2index[t_l])
            if p_l == t_l:
                correct_info[j] += 1
            total_info[j] += 1
        predict_lineages.append(cur_p_index)
        truth_lineages.append(cur_t_index)
        
    c = np.array(correct_info, dtype=np.float32)
    t = np.array(total_info, dtype=np.float32)
    r = c / t
    with open(output_path, "w") as wh:
        for p_lineage, t_lineage in zip(predict_lineages, truth_lineages):
            p_str = ",".join(list(map(lambda x: str(x), p_lineage)))
            t_str = ",".join(list(map(lambda x: str(x), t_lineage)))
            wh.write(p_str + "\t" + t_str + "\n")
    print("ACC : p__{:4f}, c__{:4f}, o__{:4f}, f__{:4f}, g__{:4f}, s__{:4f}".format(*r))
    return r

"""
build convert table for centrifuge

1. convert table build.
2. centrifuge-build -p 64 --conversion-table db_15780.table \
    --taxonomy-tree ../ncbi-tax-dump/nodes.dmp --name-table ../ncbi-tax-dump/names.dmp \
        ../GPTyrex_Data/db_15780.fa db_15780
3. 3. centrifuge -f -x db_15196_rm ../test_rm_genus_200.fasta --report-file test_rm_genus.tsv  -S test_rm_genus.txt
"""

def convertTable(fasta_path: str, output_path: str):
    with open(output_path, "w") as wh, open(fasta_path, "r") as rh:
        for line in rh:
            if ">" == line[0]:
                name = line.strip("\n")
                id_str = name.split("|")[-1]
                wh.write(name[1:] + '\t' + id_str + "\n")


def centrifuge_calculate_acc(res_file_path: str, test_fasta_path, db_seq_id_path: str, output_path: str):
    truth = []
    predict = []
    output_set = set()
    vocab2index = {}
    id2taxaid = {}
    
    with open(db_seq_id_path, "r") as rh:
        for line in rh:
            info = line.strip("\n").split("|")
            id2taxaid[info[0][1:]] = info[-1]

    c = 0
    with open(res_file_path, "r") as rh:
        for line in rh:
            if c == 0:
                c += 1
                continue
            info = line.strip("\n").split("\t")
            t = info[0]
            if t not in output_set:
                output_set.add(t)
                truth.append(int(t.split("|")[-1]))
                pre_info = info[1].split("|")
                if len(pre_info) == 1:
                    predict.append(0)
                else:
                    p_seq_id = id2taxaid[pre_info[0]]
                    predict.append(int(p_seq_id))

    ids_set = set(list(readFasta(test_fasta_path).keys()))
    for id_ele in ids_set:
        if id_ele[1:] not in output_set:
            truth.append(int(id_ele.split("|")[-1]))
            predict.append(0)
    id2lineage = getTaxoID2lineage(truth + predict, "./centrifuge_id2lineage.txt")

    correct_info = [0. for _ in range(6)]
    total_info = [0. for _ in range(6)]
    predict_lineages = []
    truth_lineages = []
    k = 0
    for p, t in zip(predict, truth):
        # print(p, t)
        p_line = id2lineage[str(p)].split(";")[1:]  # since 0 index is kingdom, start from phylum
        t_line = id2lineage[str(t)].split(";")[1:]
        for ele in p_line:
            if ele not in vocab2index:
                vocab2index[ele] = k
                k += 1
        
        for ele in t_line:
            if ele not in vocab2index:
                vocab2index[ele] = k
                k += 1
        cur_p_index = []
        cur_t_index = []
        for j, (p_l, t_l) in enumerate(zip(p_line, t_line)):
            
            cur_p_index.append(vocab2index[p_l])
            cur_t_index.append(vocab2index[t_l])
            if p_l == t_l:
                correct_info[j] += 1
            total_info[j] += 1
        predict_lineages.append(cur_p_index)
        truth_lineages.append(cur_t_index)
        
    c = np.array(correct_info, dtype=np.float32)
    t = np.array(total_info, dtype=np.float32)
    r = c / t
    with open(output_path, "w") as wh:
        for p_lineage, t_lineage in zip(predict_lineages, truth_lineages):
            p_str = ",".join(list(map(lambda x: str(x), p_lineage)))
            t_str = ",".join(list(map(lambda x: str(x), t_lineage)))
            wh.write(p_str + "\t" + t_str + "\n")
    print("ACC : p__{:4f}, c__{:4f}, o__{:4f}, f__{:4f}, g__{:4f}, s__{:4f}".format(*r))
    return r

def f1_precision_recall(y_pred: List[float], y_true: List[float],
                        average = "weighted"):
    f1 = metrics.f1_score(y_true, y_pred, average=average, zero_division=0.)
    precision = metrics.precision_score(y_true, y_pred, average=average, zero_division=0.)
    recall = metrics.recall_score(y_true, y_pred, average=average, zero_division=0.)
    acc = metrics.accuracy_score(y_true, y_pred)
    return f1, precision, recall, acc