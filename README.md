# Debiasing BERT using Partitioned Contrastive Gradient Unlearning (PCGU)

This project implements **Partitioned Contrastive Gradient Unlearning (PCGU)** to reduce gender bias in the BERT language model. We apply the method to the Winogender Schemas dataset and evaluate its effectiveness using both quantitative fairness metrics and qualitative sentence similarity tests.

##  Overview

Pretrained language models like BERT often reflect and amplify societal biases present in their training data. This project demonstrates how PCGU can be used to selectively update biased layers in BERT, reduce gender-based associations, and preserve model utility.

### Key Features
- Uses triplet contrastive loss on gendered sentence variations
- Identifies and updates only the most bias-encoding transformer layers
- Applies projection to remove gender direction from embeddings
- Evaluates bias reduction with Stereotype Score, Language Modeling Score, and ICAT

##  Project Structure

```bash
.
├── data/                   # Winogender Schemas dataset
├── images/                 # Visuals used in documentation and results
├── pcgu/                   # Core PCGU training and evaluation code
├── results/                # Evaluation metrics, plots, and logs
├── main.py                 # Entry point for training and evaluation
├── utils.py                # Helper functions for gradients, loss, etc.
├── report.tex              # LaTeX report (academic write-up)
├── bert_pcgu.png           # Example visualization of PCGU process
└── README.md               # This file
