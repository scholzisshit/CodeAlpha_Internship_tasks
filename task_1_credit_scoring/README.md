#  Task 1: Credit Scoring Model
### *"Judging people's financial decisions so banks don't have to"*

---

## What This Does
Predicts whether someone will default on a loan using their financial data.  
Translation: teaches a computer to be a judgmental bank manager.

## How to Run

```bash
# Step 1: install dependencies (one time thing)
pip install -r requirements.txt

# Step 2: (Optional) Download the actual dataset
# Go to: https://www.kaggle.com/datasets/laotse/credit-risk-dataset
# Download credit_risk_dataset.csv and put it in this folder
# OR just skip it - the script generates synthetic data automatically

# Step 3: run the script
python credit_scoring.py
```

## What Gets Generated
- `credit_scoring_results.png` → plots with confusion matrix, ROC-AUC, feature importance

## Models Used
| Model | Why |
|-------|-----|
| Random Forest | The reliable one. Handles messy data well. |
| Logistic Regression | Simple baseline. Surprisingly decent. |

## Key Metrics Explained (for the LinkedIn post)
- **Precision**: "When it says someone will default, how often is it right?"
- **Recall**: "Of all people who actually defaulted, how many did it catch?"
- **F1-Score**: Harmonic mean of precision and recall (the balanced one)
- **ROC-AUC**: Overall model quality. 1.0 = perfect. 0.5 = literally guessing.

## Features in the Dataset
| Feature | What it is |
|---------|-----------|
| `person_income` | How much $$$ they make |
| `loan_amnt` | How much they're borrowing |
| `loan_percent_income` | loan/income ratio (most important!) |
| `loan_intent` | Why they need the money |
| `loan_grade` | Credit grade A-G |
| `cb_person_default_on_file` | Have they defaulted before? (sus) |

---
*CodeAlpha Internship | Task 1 | ML Domain*
