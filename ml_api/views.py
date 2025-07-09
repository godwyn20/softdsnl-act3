# ml_api/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib

model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

class PredictView(APIView):
    def post(self, request):
        try:
            petal_length = float(request.data.get("petal_length"))
            petal_width = float(request.data.get("petal_width"))

            prediction = model.predict([[petal_length, petal_width]])
            label = label_encoder.inverse_transform(prediction)[0]

            return Response({"prediction": label})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)