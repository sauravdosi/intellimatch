# 🧠 IntelliMatch: Intelligent Company Name Matching

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/release/python-3120/) [![TensorFlow 2.19](https://img.shields.io/badge/tensorflow-2.19-orange)](https://www.tensorflow.org/) [![PyTorch 2.7](https://img.shields.io/badge/torch-2.7-red)](https://pytorch.org/) [![spaCy 3.8](https://img.shields.io/badge/spacy-3.8-green)](https://spacy.io/)  

[demo.mp4](img/demo.mp4)

---

## Table of Contents

1. [Features](#features)  
2. [Modules](#modules)  
3. [Installation](#installation)  
4. [Training](#training)  
5. [Deployment](#deployment)  
6. [Metrics](#metrics)  
7. [Conclusion](#conclusion)  
8. [License](#license)  

---

## Features

- 🚀 **Seamless Data Onboarding**  
  Effortlessly ingest JSON, CSV, or database records with one-click preprocessing pipelines.

- 🧠 **Smart Keyword Spotlight**  
  Leverage a trained classifier to highlight and color-code critical company keywords in real time.

- 🔗 **AI-Powered Fuzzy Matching**  
  Combine machine learning and fuzzy logic for ultra-accurate record linkage—even across messy data.

- 📈 **Dynamic Post-Processing & Export**  
  Auto-filter, rank, and format your final matches, then export to CSV, Excel, or your favorite BI tool.

- 📊 **Integrated Metrics Dashboard**  
  Track end-to-end performance with built-in accuracy, precision, and latency monitoring.

- 🎨 **Interactive Streamlit UI**  
  Navigate each pipeline stage via a sleek, user-friendly web interface—no coding required!

---
## 🚀 Five-Stage Workflow Overview

Instead of listing modules, we walk you through the _five key stages_ that power our end-to-end record-linkage pipeline. Each stage comes with intuitive visuals and interactive metrics to keep you in control.

---

### Stage 1: K-Fold TF-IDF Generation 🔍

![tfidf.gif](img/tfidf.gif)

In this foundational stage, we transform raw company names into rich, normalized feature vectors:

1. **Data Cleaning & Standardization**  
   - Strip punctuation, normalize case, and expand common abbreviations (e.g., “Co.” → “Company”).  
   - Ensure consistency across all records for reliable downstream matching.

2. **Smart Tokenization**  
   - Break cleaned names into meaningful tokens (e.g., “Scott Electric Co, Inc.” → [“Scott”, “Electric”, “Company”, “Inc”]).  
   - Retain semantic units like “Inc” and “Co” for context.

3. **Term Frequency Calculation**  
   - Compute raw term frequencies within each name.  
   - Visualize token distributions in intuitive bar and pie charts to spot dominant terms at a glance.  

4. **K-Fold TF-IDF Normalization**  
   - Split your data into **k** folds (default `k=5`) to prevent overfitting.  
   - Generate TF-IDF scores fold by fold, then aggregate for robust, unbiased term weights.  
   - View side-by-side heatmaps of term importance across folds.

5. **Feature Matrix Output**  
   - Produce a final, normalized TF-IDF matrix ready for both classification and fuzzy matching.  
   - Persist these features to disk or memory for lightning-fast lookup in later stages.

> **Why K-Fold?**  
> K-fold cross-validation at the TF-IDF stage ensures that term weights generalize well, giving you rock-solid features even when your company list evolves or grows.

### Stage 2: NLP Preprocessing 🤖✨

![nlp_preprocess.gif](img/nlp_preprocess.gif)

In Stage 2, we turn cleaned tokens into rich linguistic features that power both classification and matching:

1. **SpaCy Powerhouse**  
   - Load `en_core_web_lg` for fast, 300-dimensional word embeddings  
   - Load `en_core_web_trf` for transformer-powered part-of-speech tagging  
   - Seamlessly switch between models for speed vs. accuracy trade-offs

2. **Contextual Word Embeddings**  
   - Map each token (e.g. “Scott”, “Electric”, “Company”, “Inc”) to a dense 300-dim vector  
   - Capture subtle semantic relationships (e.g. “Co.” ≈ “Company”)  
   - Stack embeddings per name for downstream neural modules

3. **Part-of-Speech Tagging**  
   - Label tokens with POS tags (PROPN, ADJ, NOUN, etc.)  
   - Leverage grammatical roles to spot suffixes and corporate designators  
   - Use POS cues to refine classification and fuzzy matching logic

4. **Combined Feature Bundle**  
   - Merge embeddings + POS one-hot vectors into a single feature set  
   - Optionally serialize as NumPy arrays or Torch tensors  
   - Ready for stage 3’s keyword classifier or stage 4’s matching engine

> **Pro tip:**  
> By blending transformer accuracy with spaCy’s efficiency, you get a nimble pipeline that scales from thousands to millions of records without losing linguistic nuance.  

### Stage 3: Transformer-Powered Keyword Classification 🎯🌟

![keyword_class.gif](img/keyword_class.gif)

Now that we’ve assembled rich token features, Stage 3 uses a lightweight transformer to tag each term with its role in the company name:

1. **Multi-Feature Encoding**  
   - **Word Embeddings**: 300-dim vectors from SpaCy’s `en_core_web_lg`  
   - **Part-of-Speech One-Hots**: PROPN, NOUN, ADJ, etc., to signal corporate suffixes  
   - **TF-IDF Weights**: Normalized importance scores highlight rare but telling tokens  
   - **Positional Embeddings**: Capture token order (e.g. “Inc” at the end)

2. **Concatenate & Project**  
   - Stack embeddings, POS one-hots, TF-IDF scores, and position encodings into a unified feature tensor  
   - Feed through a compact transformer block (multi-head cross-attention + feed-forward layers)

3. **Token-Level Classification**  
   - Predict one of three labels per token:  
     - **importqnt** (pink) → truly distinctive terms like “Scott”  
     - **subsidiary** (blue) → affiliated brands or divisions  
     - **generic** (yellow) → boilerplate words like “Company” & “Inc.”  
   - Leverage context attention to spot when “Electric” is a core descriptor vs. filler

4. **Color-Coded Output**  
   - Render each token as a colored “pill” matching its label  
   - Build an intuitive legend so you always know which hue maps to which class


> **Why Transformers?**  
> Small transformers excel at sequence labeling—using attention to weigh each token’s context ensures high-precision keyword tagging, even in noisy or truncated company names.  

#### Keyword Classifier: Model Architecture 🏗️

![KW Architecture.png](img/KW%20Architecture.png)

Here’s a step-by-step breakdown of our token-level classifier, which fuses sequential, attention, and dense layers to tag each word in a company name:

1. **Multi-Modal Inputs**  
   - **Word Embeddings** (s × 300)  
   - **TF-IDF Scores** (s × 1)  
   - **Part-of-Speech One-hots** (s × n_pos)  
   - **Positional Embeddings** (s × 1)  
   > Each of these **s** token‐length arrays captures a unique signal—semantic meaning, importance, grammar, and order.

2. **Bi-LSTM Encoder**  
   - **Forward & Backward LSTMs** (64 units each direction)  
   - Learns context from both left‐to‐right and right‐to‐left sequences  
   - **BatchNorm** to stabilize and accelerate training  

3. **Self-Attention Layer**  
   - **Scaled Dot-Product Attention** (queries, keys, values all from Bi-LSTM outputs)  
   - Weighs each token’s relevance to every other token  
   - Captures long-range dependencies (e.g., linking “Inc” back to “Scott”)

4. **Forward LSTM Refinement**  
   - A single‐direction LSTM (64 units) processes the attended sequence  
   - Smooths and summarizes contextual features per token  

5. **Feature Concatenation**  
   - Stack:  
     - Attended Bi-LSTM outputs  
     - TF-IDF vector  
     - POS one-hot vector  
     - Positional embedding  
   - Forms a rich, combined feature tensor for each token

6. **Classification Head**  
   - **Dropout (p=0.1)** to prevent overfitting  
   - **Dense(64, ReLU)** → **Dense(4, ReLU)**  
   - **Softmax** over 4 classes (importqnt / subsidiary / generic / other)  
   - Outputs per‐token probabilities

> **Why this design?**  
> - **Bi-LSTM + Attention** ensures both local and global context are encoded.  
> - **Forward LSTM** refines the attention output sequentially.  
> - **Dense layers** translate complex features into crisp label probabilities.  

This hybrid architecture yields high‐precision token tags, even on noisy or unusually formatted company names.  

### Stage 4: Intelligent ML Fuzzy Matching 🔗🤖

![ml_fuzzy.gif](img/ml_fuzzy.gif)

In Stage 4, we leverage our classified keywords and encoded features to link each name to its most likely counterpart:

1. **Focused Token Filtering**  
   - Discard “generic” tokens (e.g. “Company,” “Inc.”)  
   - Retain only **importqnt** and **subsidiary** terms to sharpen comparison  

2. **Feature Vector Comparison**  
   - Assemble each company name’s key token embeddings + TF-IDF weights + POS-context  
   - Compute pairwise similarity scores via cosine distance and learned metric functions  

3. **Intelligent Fuzzy Engine**  
   - Blend classic fuzzy metrics (Levenshtein, Jaro–Winkler) with our transformer-derived features  
   - Dynamically weight each score component based on token rarity and position  
   - Output a consolidated **Match Score** (0–100%)  

4. **Categorize & Alias**  
   - **Match Category**:  
     - **Exact** (≥ 95%)  
     - **Strong** (80–94%)  
     - **Weak** (60–79%)  
     - **Manual Review** (< 60%)  
   - **Standardized Alias**: Map both variants to a canonical name (e.g., “Scott Electric Company”)

> **Key Benefit:**  
> By fusing traditional fuzzy logic with neural embeddings and context tags, Stage 4 ensures high-precision entity linking—even when names are misspelled, reordered, or abbreviated.  

### Stage 5: Postprocessing & Reporting 📑

![postprocess.gif](img/postprocess.gif)

1. **Aggregate & Filter**  
   - Group by match category, apply score thresholds, flag low-confidence cases.

2. **Dashboard & Metrics**  
   - Auto-generate pivot tables, score histograms, and precision/recall charts.

3. **Export & Deliver**  
   - Output CSV/Excel, JSON, or load directly into a database.  
   - Optionally compile a PDF/HTML summary report with top examples and metrics.


## Modules

| Module                  | Description                                                       |
| ----------------------- | ----------------------------------------------------------------- |
| `data_loader.py`        | Reads JSON/CSV, handles joins, and returns pandas DataFrames.     |
| `keyword_classifier.py` | Defines and runs the keyword classification model.                |
| `ml_fuzzy_matching.py`  | Implements fuzzy matching logic with Scikit-learn & custom rules. |
| `postprocessing.py`     | Filters, ranks, and formats final match results.                  |
| `app.py`                | Streamlit app orchestrating stages and UI.                        |
| `utils.py`              | Helper functions for logging, configuration, and visuals.         |

---

## Installation

```bash
# clone the repo
git clone https://github.com/username/repo.git
cd repo

# create and activate env
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
