# ======================================================
# 1 . Libraries
# ======================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ======================================================
# 2. Load Dataset
# ======================================================

pd.set_option('display.max_columns', 10)

df= pd.read_csv("data/UCI_Credit_Card.csv")

print("Initial shape:", df.shape)

# ======================================================
# 3. Basic Sanity Checks
# ======================================================

TARGET = "default.payment.next.month"

assert TARGET in df.columns, "Target column not found!"

df=df.drop_duplicates()

# ======================================================
# 4. Missing Values Handling (Defensive)
# ======================================================

# Separate feature types
cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
num_cols = [x for x in num_cols if x not in cat_cols]
num_cols = [c for c in num_cols if c not in [TARGET]]

# ---- Numerical Features ----
# Fill missing values with median (robust to outliers)
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# ---- Categorical Features ----
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ======================================================
# 5. Categorical Cleaning (Invalid & Rare Values)
# ======================================================
# EDUCATION: group rare/invalid value
df["EDUCATION"] = df["EDUCATION"].replace([0, 5, 6], 4)
# MARRIAGE: invalid value
df["MARRIAGE"] = df["MARRIAGE"].replace(0, 3)
for col in cat_cols:
    df[col] = df[col].astype("int")

# ======================================================
# 6. Outlier Handling (Winsorization)
# ======================================================
def outliers(series, lower=0.01, upper=0.99):
    low, high = series.quantile([lower, upper])
    return series.clip(low, high)

outlier_cols = [
    c for c in num_cols
    if c.startswith(("BILL_", "PAY_", "LIMIT_BAL"))
]
for col in outlier_cols:
    df[col] = outliers(df[col])

# ======================================================
# 7. Feature Engineering
# ======================================================
bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
pay_cols  = [f"PAY_AMT{i}" for i in range(1, 7)]
delay_cols = [f"PAY_{i}" for i in [0,2,3,4,5,6]]

# ---- Aggregations ----
df["TOTAL_BILL"] = df[bill_cols].sum(axis=1)
df["TOTAL_PAY"] = df[pay_cols].sum(axis=1)
df["PAY_RATIO"] = df["TOTAL_PAY"] / (df["TOTAL_BILL"] + 1)


# ---- Statistical Features ----
df["BILL_MEAN"] = df[bill_cols].mean(axis=1)
df["BILL_STD"]  = df[bill_cols].std(axis=1)
df["BILL_MAX"]  = df[bill_cols].max(axis=1)

df["PAY_MEAN"] = df[pay_cols].mean(axis=1)
df["PAY_STD"]  = df[pay_cols].std(axis=1)
df["PAY_MAX"]  = df[pay_cols].max(axis=1)


# ---- Behavior Features ----
df["NB_LATE_PAYMENTS"] = (df[delay_cols] > 0).sum(axis=1)

# ---- Trend Features ----
df["BILL_TREND"] = df["BILL_AMT6"] - df["BILL_AMT1"]
df["PAY_TREND"]  = df["PAY_AMT6"] - df["PAY_AMT1"]


# ======================================================
# 8. Skewness Handling (Log Transform)
# ======================================================
skewed_cols = [
    c for c in df.columns
    if c.startswith(("BILL_", "PAY_", "TOTAL_"))
]

for col in skewed_cols:
    df[col] = df[col].clip(lower=0)
    df[col] = np.log1p(df[col])


# ======================================================
# 9. Prepare Train / Test Split (No Leakage)
# ======================================================
# Drop identifier
if "ID" in df.columns:
    df=df.drop(columns=["ID"])


X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ======================================================
# 10. Final Validation
# ======================================================
print("\nFinal Train shape:", X_train.shape)
print("Final Test shape:", X_test.shape)

print("\nTrain target distribution:")
print(y_train.value_counts(normalize=True))

print("\nTest target distribution:")
print(y_test.value_counts(normalize=True))


print(df.head())
exit(0)




