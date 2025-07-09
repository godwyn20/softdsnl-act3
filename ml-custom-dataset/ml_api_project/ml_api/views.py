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