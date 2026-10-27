import pandas as pd
import joblib
import time
from sklearn.model_selection import train_test_split

model = joblib.load("model.pkl")
df = pd.read_csv("train.csv")
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
_, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Survived"])
records = test_df.iloc[:10]
for i, (_, row) in enumerate(records.iterrows(), start=1):
    prediction = int(model.predict(row[features].to_frame().T)[0])
    label = "Survived" if prediction == 1 else "Did not survive"
    print(f"Incoming record {i}/10 | PassengerId={row['PassengerId']} | Prediction={label}")
    if i < 10:
        time.sleep(1)
