# 📊 Activity 3: Build a Custom Dataset and Expose a Machine Learning API

## 🕒 Time: ~2 hours

---

## 🎯 Objectives

- Create your own labeled dataset and save it as a `.csv` file
- Load and visualize your dataset using Python
- Train a classifier using `scikit-learn`
- Save the model and label encoder to `.pkl` files
- Create a Django REST API that accepts input and returns predictions

---

## 💻 Project Folder Structure

```
ml-custom-dataset/
├── my_dataset.csv                  <-- Your custom dataset
├── train_model.py                  <-- Trains model, saves .pkl files
├── predict.py                      <-- (Optional) CLI tester for predictions
├── requirements.txt
├── README.md
├── report/                         <-- (Optional) Screenshots folder
└── ml_api_project/                 <-- Your Django project
    ├── manage.py
    ├── ml_api_project/
    │   ├── settings.py
    │   └── urls.py
    └── ml_api/
        ├── views.py
        ├── urls.py
        ├── apps.py
        └── __init__.py
```

---

## 🛠️ Part 1: Dataset and Model

### 1. Create Your Project Folder

```bash
mkdir ml-custom-dataset
cd ml-custom-dataset
```

---

### 2. Create Your Dataset in Excel or Sheets

Example (at least 2 numeric features and 1 label):

```
height,width,length,type
21.5,6.8,6.8,plastic
18.3,7.2,7.1,glass
24.0,6.0,6.2,metal
19.8,6.5,6.4,plastic
22.1,6.3,6.5,glass
17.5,5.8,5.9,metal
25.3,7.0,6.9,plastic
20.0,6.6,6.7,glass
23.7,6.9,6.8,metal
21.0,6.1,6.2,plastic
```

Save as `my_dataset.csv`.

---

### 3. Create `requirements.txt`

```txt
pandas
matplotlib
seaborn
scikit-learn
joblib
djangorestframework
```

Install everything:

```bash
pip install -r requirements.txt
```

---

### 4. Create `train_model.py`

```python
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
```

---

### 5. Create `predict.py`

```python
# predict.py

import joblib

model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")

sample = [[5.5, 2.1]]
prediction = model.predict(sample)
print("Prediction:", le.inverse_transform(prediction)[0])
```

### 6. Run `train_model.py`

```python
# train_model.py

python train_model.py
```

---

## 🌐 Part 2: Django API Setup

---

### 1. Create Django Project and App

```bash
mkdir ml_api_project
cd ml_api_project
django-admin startproject ml_api_project .
python manage.py startapp ml_api
```

---

### 2. Update `ml_api_project/settings.py`

```python
INSTALLED_APPS = [
    ...
    'rest_framework',
    'ml_api',
]
```

---

### 3. Create `ml_api/urls.py`

```python
from django.urls import path
from .views import PredictView

urlpatterns = [
    path('predict/', PredictView.as_view(), name='predict'),
]
```

---

### 4. Update `ml_api_project/urls.py`

```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('ml_api.urls')),
]
```

---

### 5. Copy model.mkl and label_encoder.pkl to ml_api folder

These files are generated in the ml-custom-dataset folder when you ran 

```python
python train_model.py
```

---

### 6. Create `ml_api/views.py`

```python
# ml_api/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib

import os
from django.conf import settings

model_path = os.path.join(settings.BASE_DIR, 'ml_api', 'model.pkl')
encoder_path = os.path.join(settings.BASE_DIR, 'ml_api', 'label_encoder.pkl')

model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)


class PredictView(APIView):
    def post(self, request):
        try:
            height = float(request.data.get("height"))
            length = float(request.data.get("length"))
            width = float(request.data.get("width"))

            prediction = model.predict([[height, width, length]])
            label = label_encoder.inverse_transform(prediction)[0]

            return Response({"prediction": label})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
```

---

### 7. Run Server

```bash
python manage.py runserver
```

Test via Postman:
- POST to `http://localhost:8000/api/predict/`
- Body (JSON):

```json
{
  "height": 21.5,
  "width": 6.8,
  "length": 6.8
}
```

---

## 📄 Final Report Format

### ✅ Submit: Report contaiting GitHub Repo + Screenshots

**Repo name:** `ml-custom-dataset`

**Expected Files:**

| File                     | Description                          |
|--------------------------|--------------------------------------|
| `my_dataset.csv`         | Your custom dataset                  |
| `train_model.py`         | Trains and saves model               |
| `predict.py`             | Optional CLI test                    |
| `ml_api_project/`        | Django project for the API           |
| `README.md`              | Final writeup and summary            |
| `report/` folder         | Screenshots (optional)               |

---

### 📷 Screenshots to Include

| Screenshot Topic              | Description                                 |
|-------------------------------|---------------------------------------------|
| 1. Raw dataset                | CSV file shown in Excel or Sheets           |
| 2. pandas preview             | `print(df.head())` from `train_model.py`    |
| 3. Visualization              | Scatterplot or seaborn output               |
| 4. Training output            | CLI print confirming model was trained      |
| 5–7. Sample predictions       | Postman or `predict.py` outputs             |
| 8–10. API response screenshots| Postman request/response to your API        |

---

### 📝 Report Should Include:

- Description of your dataset
- Features and label used
- Classifier used
- Sample inputs and predictions (Postman Screenshots)

---

## ✅ Grading Guide

| Criteria                             | Points |
|--------------------------------------|--------|
| Dataset Created and Loaded Correctly | 20     |
| Visualization with Plot              | 20     |
| Model Training                       | 20     |
| API Working with Prediction Output   | 20     |
| Organized Repo + Report              | 20     |
| **TOTAL**                            | **100**|

---
