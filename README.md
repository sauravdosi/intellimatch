# 🧠 IntelliMatch: Intelligent Company Name Matching

[![License](https://img.shields.io/github/license/sauravdosi/intellimatch)](https://opensource.org/licenses/MIT)  
[![Python Version](https://img.shields.io/pypi/pyversions/your-package)](https://www.python.org)  

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

- **Data Ingestion**: Load and preprocess JSON, CSV, or database inputs.  
- **Keyword Classification**: Classify and color-code keywords using a trained model.  
- **ML Fuzzy Matching**: Perform machine-learning based record linkage with customizable thresholds.  
- **Postprocessing & Export**: Generate final matched outputs and export to desired formats.  
- **Metrics Tracking**: Log performance metrics for training and inference.  
- **Streamlit Visualization**: Interactive UI for stepwise workflow demonstration.  

---

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
