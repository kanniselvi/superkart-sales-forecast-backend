
```python
import os
import numpy as np
import pandas as pd
import joblib

from flask import Flask, request, jsonify
from flask_cors import CORS


# ============================================================
# INITIALIZE FLASK APP
# ============================================================

superkart_api = Flask("superkart_sales_api")

CORS(superkart_api)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "superkart_sales_forecast_model_v1.joblib"
)

model = joblib.load(MODEL_PATH)


# ============================================================
# HEALTH CHECK
# ============================================================

@superkart_api.get("/")
def home():
    return "✅ Welcome to the SuperKart Sales Prediction API"


# ============================================================
# PREDICTION API
# ============================================================

@superkart_api.post("/v1/predict")
def predict_sales():

    try:

        # ----------------------------------------------------
        # Get JSON data
        # ----------------------------------------------------

        data = request.get_json()

        print("Raw incoming data:", data)


        # ----------------------------------------------------
        # Validate JSON
        # ----------------------------------------------------

        if not data:
            return jsonify({
                "error": "No JSON data received"
            }), 400


        # ----------------------------------------------------
        # Required fields
        # ----------------------------------------------------

        required_fields = [
            "Product_Weight",
            "Product_Sugar_Content",
            "Product_Allocated_Area",
            "Product_MRP",
            "Store_Size",
            "Store_Location_City_Type",
            "Store_Type",
            "Store_Age_Years",
            "Product_Type_Category"
        ]


        # ----------------------------------------------------
        # Check missing fields
        # ----------------------------------------------------

        missing_fields = [
            field
            for field in required_fields
            if field not in data
        ]

        if missing_fields:

            return jsonify({
                "error": f"Missing fields: {missing_fields}"
            }), 400


        # ----------------------------------------------------
        # Prepare model input
        # ----------------------------------------------------

        sample = {

            "Product_Weight":
                float(data["Product_Weight"]),

            "Product_Sugar_Content":
                data["Product_Sugar_Content"],

            "Product_Allocated_Area_Log":
                np.log1p(
                    float(data["Product_Allocated_Area"])
                ),

            "Product_MRP":
                float(data["Product_MRP"]),

            "Store_Size":
                data["Store_Size"],

            "Store_Location_City_Type":
                data["Store_Location_City_Type"],

            "Store_Type":
                data["Store_Type"],

            "Store_Age_Years":
                int(data["Store_Age_Years"]),

            "Product_Type_Category":
                data["Product_Type_Category"]
        }


        # ----------------------------------------------------
        # Create DataFrame
        # ----------------------------------------------------

        input_df = pd.DataFrame([sample])

        print(
            "Transformed input for model:\n",
            input_df
        )


        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        prediction = model.predict(
            input_df
        ).tolist()[0]


        # ----------------------------------------------------
        # Return prediction
        # ----------------------------------------------------

        return jsonify({
            "Predicted_Sales": float(prediction)
        })


    except Exception as e:

        print(
            "❌ Error during prediction:",
            str(e)
        )

        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    superkart_api.run(
        host="0.0.0.0",
        port=port
    )
