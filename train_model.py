# train_model.py

# --- Import Required Libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
import os

# --- 1) Load Data ---
try:
    # Make sure pima.csv is in the "data" folder inside your ai-health directory
    df = pd.read_csv("data/pima.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("FATAL ERROR: data/pima.csv not found.")
    print("➡ Please place the Pima Indians Diabetes dataset CSV in the 'data' folder.")
    exit()

# --- Handle missing/zero values (Pima dataset issue) ---
# Replace 0s in critical columns with the mean of non-zero values
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_replace:
    df[col] = df[col].replace(0, df[col].mean())

# Split features (X) and target (y)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# --- 2) Train/Test Split ---
# 80% data for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)
print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# --- 3) Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save scaler for future use
joblib.dump(scaler, "models/scaler.joblib")
print("Scaler saved in models/scaler.joblib")

# --- 4) Build Neural Network Model ---
model = models.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),  # input shape = number of features
    layers.Dense(32, activation='relu'),             # hidden layer 1
    layers.Dropout(0.2),                             # dropout to reduce overfitting
    layers.Dense(16, activation='relu'),             # hidden layer 2
    layers.Dense(1, activation='sigmoid')            # output layer (0 or 1)
])

# Compile the model (optimizer, loss function, evaluation metric)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

# Show model architecture
model.summary()

# --- 5) Train the Model ---
print("\n--- Starting Model Training (50 Epochs) ---")
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.1,  # keep 10% of training data for validation
    epochs=50,
    batch_size=16,
    verbose=2
)
print("✅ Training complete.")

# --- 6) Evaluate Model ---
print("\n--- Model Evaluation on Test Data ---")
raw_preds = model.predict(X_test_scaled, verbose=0)     # raw probabilities
preds = (raw_preds > 0.5).astype(int).ravel()           # convert to 0/1

print("Accuracy:", accuracy_score(y_test, preds))
print("AUC:", roc_auc_score(y_test, raw_preds))
print("\nClassification Report:\n", classification_report(y_test, preds))


# FIX: Added .h5 extension for reliable saving
model.save("models/diabetes_model.h5") 
print("\n✅ Model and Scaler saved successfully in the 'models/' folder.")
