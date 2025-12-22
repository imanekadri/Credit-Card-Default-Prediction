
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df= pd.read_csv("data/UCI_Credit_Card.csv")
print("Initial shape:", df.shape)

TARGET = "default.payment.next.month"

assert TARGET in df.columns, "Target column not found!"

df.drop_duplicates(inplace=True)

num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]

num_cols = [c for c in num_cols if c not in [TARGET]]
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
df["EDUCATION"] = df["EDUCATION"].replace([0, 5, 6], 4)
df["MARRIAGE"] = df["MARRIAGE"].replace(0, 3)
for col in cat_cols:
    df[col] = df[col].astype("int")
def winsorize(series, lower=0.01, upper=0.99):
    low, high = series.quantile([lower, upper])
    return series.clip(low, high)

outlier_cols = [
    c for c in num_cols
    if c.startswith(("BILL_", "PAY_", "LIMIT_BAL"))
]
for col in outlier_cols:
    df[col] = winsorize(df[col])
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

skewed_cols = [
    c for c in df.columns
    if c.startswith(("BILL_", "PAY_", "TOTAL_"))
]

for col in skewed_cols:
    df[col] = np.log1p(df[col])

# Drop identifier
if "ID" in df.columns:
    df.drop(columns=["ID"], inplace=True)

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
#  Final Validation
# ======================================================
print("\nFinal Train shape:", X_train.shape)
print("Final Test shape:", X_test.shape)

print("\nTrain target distribution:")
print(y_train.value_counts(normalize=True))

print("\nTest target distribution:")
print(y_test.value_counts(normalize=True))
