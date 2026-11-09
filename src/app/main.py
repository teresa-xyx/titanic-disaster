# %%
# --- 13 ---
print("\n" + "=" * 60)
print("Problem 13")
print("=" * 60)

from pathlib import Path
import pandas as pd

# Data folder: src/data, located relative to this script
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

train = pd.read_csv(DATA_DIR / "train.csv")
#print("Problem 13: load train.csv")
#print(f"train.csv loaded: {train.shape[0]} rows, {train.shape[1]} columns")
print(train.head())

# %%
# --- 14a. Explore all three files ---
print("\n" + "=" * 60)
print("Problem 14")
print("=" * 60)

for name in ["train.csv", "test.csv", "gender_submission.csv"]:
    df = pd.read_csv(DATA_DIR / name)
    print(f"\n===== {name}: {df.shape[0]} rows, {df.shape[1]} columns =====")
    print(pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "n_unique": df.nunique(),
    }))

print(f"\n[Explore] Overall survival rate in train: {train['Survived'].mean():.1%}")
print("\n[Explore] Survival rate by sex:")
print(train.groupby("Sex")["Survived"].mean().round(3))
print("\n[Explore] Survival rate by passenger class:")
print(train.groupby("Pclass")["Survived"].mean().round(3))

# %%
TITLE_MAP = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
COMMON_TITLES = ["Mr", "Mrs", "Miss", "Master"]

def prepare(df, label, fill=None):
    """Add features, fill missing values, and encode columns (Problem 14 steps)."""
    df = df.copy()
    print(f"\nPreparing {label} data: {df.shape[0]} rows")

    # --- 14b. Add new features ---
    # Title from Name, e.g. "Braund, Mr. Owen Harris" -> "Mr"
    df["Title"] = df["Name"].str.extract(r",\s*([^.]+)\.", expand=False).str.strip()
    df["Title"] = df["Title"].replace(TITLE_MAP)
    df["Title"] = df["Title"].where(df["Title"].isin(COMMON_TITLES), "Rare")
    print(f"\n[{label}] Title extracted from Name (rare titles grouped as 'Rare'):")
    print(df["Title"].value_counts())

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    print(f"\n[{label}] FamilySize = SibSp + Parch + 1")
    print(df["FamilySize"].describe().round(2))

    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    print(f"\n[{label}] IsAlone: {df['IsAlone'].sum()}")

    df["HasCabin"] = df["Cabin"].notna().astype(int)
    print(f"[{label}] HasCabin: {df['HasCabin'].sum()} of {len(df)} passengers")

    # --- 14c. Fill missing values (values learned from train only) ---
    if fill is None:
        fill = {
            "age_by_title": df.groupby("Title")["Age"].median(),
            "embarked": df["Embarked"].mode()[0],
            "fare": df["Fare"].median(),
        }

    n_missing_age = df["Age"].isna().sum()
    df["Age"] = df["Age"].fillna(df["Title"].map(fill["age_by_title"]))
    print(f"\n[{label}] Filled {n_missing_age} missing Age values with the median age of each Title (from train):")
    print(fill["age_by_title"].round(1))

    n_missing_emb = df["Embarked"].isna().sum()
    df["Embarked"] = df["Embarked"].fillna(fill["embarked"])
    print(f"\n[{label}] Filled {n_missing_emb} missing Embarked values with the most common port: {fill['embarked']}")

    n_missing_fare = df["Fare"].isna().sum()
    df["Fare"] = df["Fare"].fillna(fill["fare"])
    print(f"[{label}] Filled {n_missing_fare} missing Fare values with the median fare from train: {fill['fare']:.2f}")

    # --- 14d. Encode and drop columns ---
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    print(f"\n[{label}] Encoded Sex: male = 0, female = 1")

    drop_cols = ["Name", "Ticket", "Cabin"]
    df = df.drop(columns=drop_cols)
    print(f"[{label}] Dropped columns: {drop_cols}")
    print(f"[{label}] Remaining missing values: {df.isna().sum().sum()}")

    return df, fill

# %%
train, fill = prepare(train, "Train")
print(train.head())

# %%
#--- 15 ---
print("\n" + "=" * 60)
print("Problem 15")
print("=" * 60)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def to_features(df, feature_cols=None):
    # One-hot encode the remaining text columns
    X = pd.get_dummies(df, columns=["Title", "Embarked"], drop_first=True, dtype=int)

    # Features: everything except the ID, the target, and SibSp/Parch (already combined in FamilySize) and Sex(can tell from Title)
    if feature_cols is None:
        feature_cols = [c for c in X.columns if c not in ["PassengerId", "Survived", "SibSp", "Parch","Sex"]]
    return X.reindex(columns=feature_cols, fill_value=0), feature_cols


X_train,feature_cols = to_features(train)
y_train = train["Survived"]
print(f"[Model] {len(feature_cols)} features used: {feature_cols}")

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
coefs = pd.Series(model[-1].coef_[0], index=feature_cols).sort_values()
print("\nModel: Coefficients (on standardized features; positive = higher survival odds):")
print(coefs.round(3))
print(f"Model: Intercept: {model[-1].intercept_[0]:.3f}")

# %%
#--- 16 ---
print("\n" + "=" * 60)
print("Problem 16")
print("=" * 60)

from sklearn.metrics import accuracy_score, confusion_matrix

train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
print(f"[Train] Accuracy: {train_acc:.4f} ({(train_pred == y_train).sum()} of {len(y_train)} correct)")

# Baselines for comparison
majority_acc = (y_train == 0).mean()
gender_acc = ((train["Sex"] == 1).astype(int) == y_train).mean()
print(f"\n[Train] Baseline - predict everyone died:{majority_acc:.4f}")
print(f"[Train] Baseline - women survive, men die:{gender_acc:.4f}")


# %%
#--- 17 ---
print("\n" + "=" * 60)
print("Problem 17")
print("=" * 60)
test = pd.read_csv(DATA_DIR / "test.csv")
print(f"[Test] test.csv loaded: {test.shape[0]} rows, {test.shape[1]} columns")
print(f"[Test] Missing values before cleaning: {test.isna().sum()[test.isna().sum() > 0].to_dict()}")

# Same steps as Problem 14, using the fill values learned from train
test, _ = prepare(test, "Test", fill)

# Same features as Problem 15, in the same order
X_test, _ = to_features(test, feature_cols)
print(f"\n[Test] Features aligned to the {len(feature_cols)} training features")

test_pred = model.predict(X_test)
predictions = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_pred})
print(f"[Test] Predicted {len(predictions)} passengers: "
      f"{predictions['Survived'].sum()} survived ({predictions['Survived'].mean():.1%}), "
      f"{(predictions['Survived'] == 0).sum()} died")
print(predictions.head(10))

# %%
# --- 18 ---
# %%
print("\n" + "=" * 60)
print("Problem 18: Predictions on the test set")
print("=" * 60)

print("       Printing the predictions for all test passengers instead.\n")
print(predictions.to_string(index=False))
print(f"\n[Test] Total: {len(predictions)} passengers, "
      f"{predictions['Survived'].sum()} predicted to survive, "
      f"{(predictions['Survived'] == 0).sum()} predicted to die")

# %%