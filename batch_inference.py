import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

model = joblib.load("model.pkl")
df = pd.read_csv("train.csv")
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
_, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Survived"])
records = test_df.iloc[:20].copy()
result = records[["PassengerId"] + features].copy()
result["PredictedSurvived"] = model.predict(records[features]).astype(int)
result["PredictionLabel"] = result["PredictedSurvived"].map({1: "Survived", 0: "Did not survive"})
result.to_csv("batch_predictions.csv", index=False)
print(result.to_string(index=False))
print("\\nSaved 20 predictions to batch_predictions.csv")
