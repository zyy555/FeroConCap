# FeroConCap: Ferroptosis-Related Proteins Identification

---
## Overview
This repository contains the official PyTorch implementation of **FeroConCap**, a novel deep learning framework that integrates capsule networks with supervised contrastive learning for accurate identification of ferroptosis-related proteins from sequences.

> **"Contrastive Representation Learning and Capsule Networks Enable Accurate Identification of Ferroptosis-Related Proteins"**  
> *Manuscript under review*

🌐 **Web Server:** https://ycclab.cuhk.edu.cn/FeroConCap

---
## Key Features

- 🧬 **Capsule Network Integration** — Hierarchical feature learning for capturing complex protein sequence patterns
- 🎯 **Supervised Contrastive Learning** — Enhanced discriminative representations for improved classification performance  
- ⚡ **Automatic Feature Extraction** — End-to-end learning without manual feature engineering

---
## Data

The dataset is organized into two main directories:
```
data/
├── Fasta/ # Raw protein sequences
└── FCGR/ # Transformed FCGR features
```

### 1. Fasta Directory

Contains protein sequences in FASTA format:

| File | Description | Format |
|------|-------------|--------|
| `training.fasta` | Training set protein sequences | FASTA |
| `testing.fasta` | Testing set protein sequences | FASTA |

### 2. FCGR Directory

Contains **Frequency Chaos Game Representation (FCGR)** transformed data:

| File | Description | Source |
|------|-------------|--------|
| `train_fcgr.txt` | FCGR features for training | `training.fasta` |
| `test_fcgr.txt` | FCGR features for testing | `testing.fasta` |

---
## Setup and Installation
---
## Usage
