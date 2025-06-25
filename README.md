# 🧠 IntelliMatch: Intelligent Company Name Matching

[![License](https://img.shields.io/github/license/sauravdosi/intellimatch)](https://opensource.org/licenses/MIT)  
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/release/python-3120/)  
[![TensorFlow 2.19](https://img.shields.io/badge/tensorflow-2.19-orange)](https://www.tensorflow.org/)  
[![PyTorch 2.7](https://img.shields.io/badge/torch-2.7-red)](https://pytorch.org/)  
[![spaCy 3.8](https://img.shields.io/badge/spacy-3.8-green)](https://spacy.io/)  

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
