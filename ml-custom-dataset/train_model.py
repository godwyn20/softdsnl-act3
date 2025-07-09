# train_model.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
df = pd.read_csv("my_dataset.csv")

# Plot the data
sns.pairplot(df, hue="type", diag_kind="kde")
plt.suptitle("Bottle Dimensions by Type", y=1.02)
plt.show()

# Features and target
X = df[["height", "width", "length"]]
le = LabelEncoder()
y = le.fit_transform(df["type"])

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model and label encoder
joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("✅ Model trained and saved.")


