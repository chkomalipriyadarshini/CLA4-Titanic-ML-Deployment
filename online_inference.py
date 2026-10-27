import pandas as pd
import joblib

model = joblib.load("model.pkl")
# One held-out record from train.csv is used only as an inference input.
# It is excluded from model fitting by selecting from the test portion.
df = pd.read_csv("train.csv")
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
from sklearn.model_selection import train_test_split
_, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Survived"])
passenger = test_df.iloc[[0]][features]
prediction = int(model.predict(passenger)[0])
print("Passenger input:")
print(passenger.to_string(index=False))
print("Prediction:", "Survived" if prediction == 1 else "Did not survive")
