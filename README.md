# TaCoGPT

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

## Description

TaCoGPT is a large language model employed for the taxonomic classification of microbial sequences.

***WARNING***: This work is still in progress.

## TODO:
Write interface.

## How to run
Pre-train model

```bash
python script_pretrain_TaCoGPT_ddp.py
```

Fine-tune model with all data:
```bash
python script_train_TaCoGPT_ddp_All.py
```

Fine-tune model with removed clades data.
```bash
python script_train_TaCoGPT_ddp_Rm.py
```


