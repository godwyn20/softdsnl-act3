# train_model.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("my_dataset.csv")

sns.scatterplot(data=df, x="petal_length", y="petal_width", hue="species")
plt.title("Custom Dataset")
plt.show()

X = df[["petal_length", "petal_width"]]
le = LabelEncoder()
y = le.fit_transform(df["species"])

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("✅ Model trained and saved.")