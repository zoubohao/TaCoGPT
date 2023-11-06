

<div align="center">

# TaCoGPT

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

</div>

## Description

TaCoGPT is a large language model employed for the taxonomic classification of microbial sequences.

## TODO:
Write interface function to do training and inference.

## How to run

Install the requirements for the requirements.txt file.

---------------------------------------------------------

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


