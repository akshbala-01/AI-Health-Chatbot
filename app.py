# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import tensorflow as tf
import shap
import os

app = Flask(__name__)

# --- CONFIGURATION ---
# Define expected features in order
FEATURES = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']

# --- MODEL AND SCALER LOADING ---
scaler = None
model = None
explainer = None
X_train_scaled_sample = None
feature_names = None

try:
    # 1. Load scaler
    scaler_path = "models/scaler.joblib"
    if not os.path.exists(scaler_path): raise FileNotFoundError(f"{scaler_path} not found.")
    scaler = joblib.load(scaler_path)
    
    # 2. Load Keras model - USING THE CORRECT .h5 FILE FORMAT
    model_path = "models/diabetes_model.h5"
    if not os.path.exists(model_path): raise FileNotFoundError(f"{model_path} not found.")
    model = tf.keras.models.load_model(model_path)
    
    print("Model and Scaler loaded successfully.")

    # 3. Load SHAP components (optional, may fail if not created)
    shap_path = "models/shap_components.joblib"
    if os.path.exists(shap_path):
        shap_data = joblib.load(shap_path)
        X_train_scaled_sample = shap_data['X_train_scaled_sample']
        feature_names = shap_data['feature_names']
        
        # 4. Initialize SHAP explainer
        explainer = shap.KernelExplainer(model.predict, X_train_scaled_sample[:10]) 
        print("SHAP explainer loaded successfully.")
    else:
        print("WARNING: shap_components.joblib not found. Running without explainability.")
    
except FileNotFoundError as fnfe:
    print(f"ERROR: A required model file was not found: {fnfe}")
except Exception as e:
    # This catches complex errors like SHAP initialization or TensorFlow issues
    print(f"ERROR: Failed to load components due to a runtime error: {e}")
    
# --- ROUTES ---

@app.route("/")
def home():
    """Serves the Chatbot HTML page."""
    # The return render_template('index.html') relies on the 'templates' folder
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    """Handles the prediction request from the chatbot."""
    # Check if essential components are loaded
    if not model or not scaler:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500
        
    data = request.json
    
    # 1. Prepare input data
    x = [float(data.get(f, 0)) for f in FEATURES]
    x_arr = np.array([x])
    x_scaled = scaler.transform(x_arr)
    
    # 2. Get prediction
    prob = float(model.predict(x_scaled, verbose=0)[0][0])
    risk = "High" if prob >= 0.7 else ("Medium" if prob >= 0.4 else "Low")

    # 3. Calculate SHAP values for explanation
    explanation = []
    if explainer:
        try:
            # Calculate SHAP values for the current input
            shap_values = explainer.shap_values(x_scaled, silent=True)
            
            for i, (name, value) in enumerate(zip(feature_names, shap_values[0][0])):
                contribution = "pushing risk UP" if value > 0 else "pushing risk DOWN"
                
                explanation.append({
                    "feature": name,
                    "value": x_arr[0][i],
                    "contribution_score": round(float(value), 3),
                    "impact": contribution
                })
                
            explanation.sort(key=lambda item: abs(item['contribution_score']), reverse=True)
            
        except Exception as e:
            print(f"ERROR calculating SHAP values: {e}")
            explanation.append({"error": "Failed to calculate explanation."})


    return jsonify({
        "probability": prob, 
        "risk": risk,
        "explanation": explanation[:3],
        "features_used": data
    })

# --- RUN APPLICATION ---

if __name__ == "__main__":
    if model and scaler:
        app.run(debug=True, port=5000)
    else:
        print("Application stopped because Model or Scaler failed to load. Please check logs for file not found errors.")