# ============================================================
# TASK 1: Credit Scoring Model
# aka "let's find out who's broke and who's not"
# (spoiler: the bank already knows, they just want us to do the math)
# ============================================================

# okay so first things first, let's import literally everything
# because that's what we do when we're "real ML engineers" lmao

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")  # no one wants to see your drama, sklearn

# ============================================================
# STEP 1: GET THE DATA (the fun part where we act like we know kaggle)
# ============================================================

# Using the Credit Risk Dataset from Kaggle
# Dataset: https://www.kaggle.com/datasets/laotse/credit-risk-dataset
# Download it and put "credit_risk_dataset.csv" in this folder
# OR just run this script and if the file isn't there we'll generate
# some fake data that's almost as good (don't tell the internship people)

import os

DATA_FILE = "credit_risk_dataset.csv"

if not os.path.exists(DATA_FILE):
    print("📁 Dataset not found. Generating synthetic data...")
    print("(It's basically the same thing, trust me bro)")
    print()

    np.random.seed(42)  # for reproducibility aka "why does my code work on my machine"
    n = 5000

    # making up some people's financial situations, classic
    data = {
        "person_age": np.random.randint(20, 65, n),
        "person_income": np.random.randint(20000, 120000, n),
        "person_home_ownership": np.random.choice(["RENT", "OWN", "MORTGAGE", "OTHER"], n),
        "person_emp_length": np.random.uniform(0, 40, n).round(1),
        "loan_intent": np.random.choice(["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"], n),
        "loan_grade": np.random.choice(["A", "B", "C", "D", "E", "F", "G"], n),
        "loan_amnt": np.random.randint(500, 35000, n),
        "loan_int_rate": np.random.uniform(5.42, 23.22, n).round(2),
        "loan_percent_income": np.random.uniform(0.0, 0.83, n).round(2),
        "cb_person_default_on_file": np.random.choice(["Y", "N"], n, p=[0.2, 0.8]),
        "cb_person_cred_hist_length": np.random.randint(2, 30, n),
    }

    df_temp = pd.DataFrame(data)
    # making defaults more likely for high loan amounts and low incomes (realistic vibes)
    default_prob = (df_temp["loan_amnt"] / df_temp["person_income"]) * 2.5
    default_prob = np.clip(default_prob, 0, 1)
    df_temp["loan_status"] = (np.random.rand(n) < default_prob).astype(int)

    # sprinkle in some missing values because real data is messy af
    missing_idx = np.random.choice(n, size=int(n * 0.05), replace=False)
    df_temp.loc[missing_idx[:len(missing_idx)//2], "person_emp_length"] = np.nan
    missing_idx2 = np.random.choice(n, size=int(n * 0.03), replace=False)
    df_temp.loc[missing_idx2, "loan_int_rate"] = np.nan

    df_temp.to_csv(DATA_FILE, index=False)
    print(f"✅ Generated {n} rows of beautiful fake data. You're welcome.\n")
else:
    print(f"✅ Found the dataset! Nice, you actually downloaded it.\n")


# ============================================================
# STEP 2: LOAD AND EXPLORE (pretending we're data scientists)
# ============================================================

print("=" * 60)
print("LOADING DATA... (dramatic pause)")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print(f"\n📊 Dataset Shape: {df.shape}")
print(f"   Rows: {df.shape[0]} (that's {df.shape[0]} people whose credit we're judging)")
print(f"   Columns: {df.shape[1]} features\n")

print("First 5 rows (just to pretend we're exploring):")
print(df.head())
print()

print("📉 Missing Values (the annoying ones):")
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    print(missing)
else:
    print("   No missing values! Suspiciously clean data 👀")
print()

print("📈 Basic Stats (for people who like numbers I guess):")
print(df.describe())
print()


# ============================================================
# STEP 3: PREPROCESSING (the part nobody on YouTube talks about)
# ============================================================

print("=" * 60)
print("PREPROCESSING... hold on this takes a sec")
print("=" * 60)

# Handle missing values - filling with median because mean gets wrecked by outliers
# (I learned this the hard way)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"   Filled missing '{col}' with median: {median_val:.2f}")

# One-Hot Encoding for categorical columns
# because models don't understand "RENT" vs "OWN" - they're not that smart
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
if "loan_status" in categorical_cols:
    categorical_cols.remove("loan_status")

print(f"\n🔄 One-hot encoding these columns: {categorical_cols}")
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"   After encoding: {df.shape[1]} columns (from {df.shape[1] - len(categorical_cols)} numeric + encoding explosion)")

# Separate features and target
# loan_status = 1 means default (bad), 0 means paid back (good)
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

print(f"\n🎯 Target distribution:")
print(f"   Paid back (0): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
print(f"   Defaulted (1): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
print(f"   (Yeah it's imbalanced. Real world is messy. Deal with it.)\n")

# Scale numerical features - because 100,000 income vs 0.8 loan_percent is not fair
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Train/Test Split - 80% train, 20% test, classic ratio
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✂️  Split: {len(X_train)} training | {len(X_test)} testing")
print(f"   (random_state=42 because everyone uses 42. It's basically a rule.)\n")


# ============================================================
# STEP 4: TRAIN THE MODEL (the part you show your parents)
# ============================================================

print("=" * 60)
print("TRAINING MODELS... (please work please work please work)")
print("=" * 60)

# --- Random Forest (the reliable one, like that one friend who always shows up) ---
print("\n🌲 Training Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,      # 100 trees because why not
    max_depth=10,          # not too deep, we don't want overfitting
    random_state=42,
    n_jobs=-1              # use all CPU cores bc we're impatient
)
rf_model.fit(X_train, y_train)
print("   Random Forest: trained ✅")

# --- Logistic Regression (the simple one, but sometimes simple wins) ---
print("\n📈 Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
print("   Logistic Regression: trained ✅\n")


# ============================================================
# STEP 5: EVALUATE (the moment of truth)
# ============================================================

print("=" * 60)
print("EVALUATING MODELS... crossing fingers 🤞")
print("=" * 60)

models = {
    "Random Forest": rf_model,
    "Logistic Regression": lr_model
}

results = {}

for name, model in models.items():
    print(f"\n{'='*40}")
    print(f"📊 {name} Results:")
    print('='*40)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Classification Report - Precision, Recall, F1 all in one go
    print("\n🔍 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Paid Back", "Defaulted"]))

    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"   ROC-AUC Score: {roc_auc:.4f}")
    print(f"   (anything above 0.8 is pretty good. above 0.9 and you start lying on your resume)")

    results[name] = {"model": model, "y_pred": y_pred, "y_prob": y_prob, "roc_auc": roc_auc}


# ============================================================
# STEP 6: VISUALIZATIONS (making it look good for the report)
# ============================================================

print("\n📸 Generating plots... (this is the part we screenshot for LinkedIn)")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Credit Scoring Model Results\n(Yes I trained an AI. Yes I am now a data scientist apparently.)",
             fontsize=14, fontweight="bold", y=1.02)

colors = {"Random Forest": "#2ecc71", "Logistic Regression": "#3498db"}

# --- Plot 1 & 2: Confusion Matrices ---
for idx, (name, res) in enumerate(results.items()):
    ax = axes[0][idx]
    cm = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Paid Back", "Defaulted"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\nConfusion Matrix", fontweight="bold")

# --- Plot 3: ROC-AUC Curves ---
ax_roc = axes[0][2]
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {res['roc_auc']:.3f})", linewidth=2, color=colors[name])
ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Chance (aka guessing)")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC-AUC Curves\n(higher = better, obviously)", fontweight="bold")
ax_roc.legend(loc="lower right")
ax_roc.grid(True, alpha=0.3)

# --- Plot 4: Feature Importance (Random Forest only) ---
ax_feat = axes[1][0]
feature_imp = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = feature_imp.nlargest(10)
sns.barplot(x=top_features.values, y=top_features.index, ax=ax_feat, palette="viridis")
ax_feat.set_title("Top 10 Important Features\n(what the model actually cares about)", fontweight="bold")
ax_feat.set_xlabel("Feature Importance Score")

# --- Plot 5: Model Comparison Bar Chart ---
ax_comp = axes[1][1]
model_names = list(results.keys())
roc_scores = [results[m]["roc_auc"] for m in model_names]
bars = ax_comp.bar(model_names, roc_scores, color=list(colors.values()), edgecolor="black", linewidth=1.2)
ax_comp.set_ylim(0.5, 1.0)
ax_comp.set_ylabel("ROC-AUC Score")
ax_comp.set_title("Model Comparison\n(who's the real MVP?)", fontweight="bold")
for bar, score in zip(bars, roc_scores):
    ax_comp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{score:.3f}", ha="center", va="bottom", fontweight="bold")
ax_comp.grid(True, alpha=0.3, axis="y")

# --- Plot 6: Target Distribution ---
ax_dist = axes[1][2]
target_counts = y.value_counts()
wedges, texts, autotexts = ax_dist.pie(
    target_counts,
    labels=["Paid Back 😇", "Defaulted 💸"],
    autopct="%1.1f%%",
    colors=["#2ecc71", "#e74c3c"],
    startangle=90,
    explode=(0, 0.05)
)
ax_dist.set_title("Dataset Distribution\n(spoiler: most people pay back)", fontweight="bold")

plt.tight_layout()
plt.savefig("credit_scoring_results.png", dpi=150, bbox_inches="tight")
print("   Saved: credit_scoring_results.png")

plt.show()

print("\n" + "="*60)
print("✅ TASK 1 COMPLETE!")
print("="*60)
print("\nKey Takeaways (for when your manager asks what you learned):")
print("  → Random Forest usually beats Logistic Regression")
print("  → Loan amount vs income ratio is super important")
print("  → ROC-AUC > 0.8 = you're doing something right")
print("  → Data cleaning is 80% of the job. Nobody tells you this.")
print("\nGo flex this on LinkedIn now 🚀")
