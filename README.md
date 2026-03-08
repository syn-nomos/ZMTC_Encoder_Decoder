# ZMTC Encoder–Decoder

Official repository for the **22nd International Conference on Artificial Intelligence Applications and Innovations (AIAI)** paper:

**"Enhancing Zero-shot Multi-label Text Classification with LLM-generated Label Descriptions and Encoder–Decoder Models"**

# Abstract
Zero-shot multi-label text classification is the task of assigning
all relevant labels to an input text when there are no training instances
for at least some of the available labels. This task is important
in application domains where the set of labels is constantly enriched and
it is not feasible to obtain adequate training data for new categories,
e.g., news, legal documents, healthcare, and e-commerce. A common approach
in this area is to attempt to represent both documents and label
descriptions in a common embedding space to reveal the most relevant
labels per document. To this end, very short label descriptions, often a single word or a few words, are used. Moreover, there is an inherent difficulty in predicting the appropriate number of labels per document. In this paper, we consider the use of LLM-generated label descriptions, allowing existing zero-shot multi-label text classification methods to better represent labels and estimate their relevance to documents. In addition, we propose an encoder-decoder approach that can adapt the number of predicted labels for each input document. In the presented experiments, we used two datasets from the legal and healthcare domains to demonstrate how the performance of existing methods is improved when LLM-generated label descriptions are used. We also test various encoder-decoder models and report improved performance results.

---

# Generative Multi-Label Classification with Pointer Decoder

This repository implements a **generative multi-label text classification model** using:

- Transformer **document encoder** (e.g., XLM-R, DistilBERT)
- **T5-style decoder**
- **Pointer mechanism** over label embeddings
- **EOS stopping token**
- **Count head** for label cardinality bias

The model predicts labels **sequentially** and stops generation using an **EOS token**, making it suitable for tasks such as:

- **EURLEX57K**
- **MIMIC-III**

multi-label classification.

---

# Project Structure

By default, the repository expects the following structure relative to the root directory:
```text
ZMTC_Encoder_Decoder/
├─ src
    ├── main.py          # Main training script
    ├── model.py         # Encoder + T5 pointer decoder model
    ├── dataset.py       # JSONL dataset loader and collator
    ├── utils.py         # Helper functions (label loading, label memory etc.)
    ├── train.py         # Training loop
    ├── eval.py          # Evaluation pipeline
    ├── metrics.py       # Multi-label metrics
    ├── config.json      # Example configuration file
```

## Architecture Overview

The model is implemented in model.py and consists of:

- **Encoder**

- **HuggingFace transformer** (e.g. XLM-Roberta, DistilBERT)

- **Supports chunked document encoding for long documents**

- **Optional freezing or partial fine-tuning**

- **Decoder**

- **T5Stack decoder (not pretrained)**

- **Takes previously generated label embeddings as input**

- **Pointer Head**

Instead of predicting tokens from a vocabulary, the decoder points to:
```text [label_1, label_2, ..., label_L, EOS] ```
using a dot-product with precomputed label embeddings.

## Environment Setup
Requires Python 3.12+
### 1. Create a Virtual Environment

```bash
python -m venv .venv
```
Activate it:

For Linux / macOS
```bash
source .venv/bin/activate
```
For Windows
```bash
.venv\Scripts\activate
```
### Install Dependencies
```bash
pip install -U pip
pip install -r requirements.txt
```

The dataset should be stored in JSONL format.
### Configuration

Training parameters and dataset paths are defined in config.json. You must set the paths before execution.

### Running the Experiment
```bash
python main.py --path config.json
```
This will:

1. Load dataset

2. Build label embeddings

3. Initialize model

4. Train using teacher forcing

5. Evaluate on validation set

6. Save checkpoints

Output directory:
```text
ZMTC_Encoder_Decoder/
├─ src
    output/
           best.pt
           last.pt
           history.json 
```
Using OFLAN Weights

If you want to initialize the encoder with OFLAN pretrained weights:
```bash
python main.py --path config.json --load-oflan
```

## OF-LAN

Details for experiments are available at [Preserving Zero-shot Capability in Supervised Fine-tuning for Multi-label
Text Classification](https://www.csie.ntu.edu.tw/~cjlin/papers/zero_shot_one_side_tuning/)

## RTS
Details for experiments are available at ["Structural Contrastive Representation Learning for Zero-shot Multi-label Text Classification" in Findings of EMNLP 2022](https://github.com/tonyzhang617/structural-contrastive-representation-learning)


## Datasets

### EURLEX57K
EURLEX57K could be downloaded from [EURLEX57K](https://huggingface.co/datasets/jonathanli/eurlex)
