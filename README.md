# Home Credit Default Risk — EDA + Baseline Models

This project explores the **Home Credit** loan application dataset and builds baseline machine learning models to predict **default risk** (`TARGET`).
It includes data cleaning, exploratory analysis with visualizations, and comparison of multiple classifiers.

---

## Overview

- Dataset: `data_train.csv`
- Rows: ~307k
- Columns used in this notebook: 33 (subset of full Home Credit dataset)
- Target: `TARGET`
  - `0` = no default
  - `1` = default (higher risk)

---

## What’s Inside

### 1) Data Inspection
- `head()`, `info()`, missing values check (`isnull().sum()`)

### 2) Data Cleaning
- Dropped rows with missing values in:
  - `AMT_ANNUITY`, `AMT_GOODS_PRICE`, `NAME_TYPE_SUITE`, `EXT_SOURCE_2`
- Filled missing `OCCUPATION_TYPE` using **mode** (most frequent value)

---

## Exploratory Data Analysis (EDA)

Visualizations included:

- Gender distribution (pie chart)
- Gender count by occupation type (count plot)
- Organization type vs target (count plot)
- Credit amount distribution (hist + KDE)
- Average income by occupation type (bar plot)
- Credit distribution by target (violin plot)
- Contract type vs target (count plot)
- Average target rate by income type (bar plot)
- Education type distribution
- Family status distribution

---

## Modeling (Baseline Comparison)

Models trained and compared using a simple feature set:

**Features used:**
- `AMT_CREDIT`
- `AMT_ANNUITY`
- `AMT_GOODS_PRICE`
- `AMT_INCOME_TOTAL`

**Models:**
- Random Forest
- Decision Tree
- Support Vector Machine (RBF kernel + StandardScaler)

**Evaluation:**
- Train/test split (80/20)
- Accuracy score
- Classification report (precision, recall, f1-score)
- Best model selection by:
  - Accuracy
  - F1 score

> Note: The dataset is highly imbalanced (TARGET=1 is much rarer), so **accuracy can be misleading**. F1/recall for class `1` is important.

---

## Results Summary (Example Run)

- Random Forest Accuracy: ~0.91
- Decision Tree Accuracy: ~0.89
- SVM Accuracy: ~0.92 (but predicted almost all as class 0)

Best model by **accuracy**: Random Forest  
Best model by **F1** (class 1): Decision Tree (in this run)

---

## How to Run

### 1) Install dependencies
```bash
pip install -r requirements.txt
```
```bash
Create requirements.txt:

numpy
pandas
matplotlib
seaborn
scikit-learn
```
