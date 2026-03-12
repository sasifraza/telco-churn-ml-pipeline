import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Load data
path = "../data/Telco_Customer_Churn.csv"
df = pd.read_csv(path)

# 2. Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# 3. Encode target
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# 4. Drop ID
df = df.drop("customerID", axis=1)

# 5. One-hot encode
df = pd.get_dummies(df, drop_first=True).astype(int)

# 6. Split
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 8. Save model
joblib.dump(model, "../models/random_forest.pkl")

print("Model trained and saved.")

