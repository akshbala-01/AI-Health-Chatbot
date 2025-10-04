# 🩺 AI Health Chatbot (Diabetes Risk Predictor)

This is a full-stack machine learning web application that predicts the risk of Type 2 Diabetes based on clinical inputs.

## ✨ Key Features
* **Model:** Sequential Neural Network (TensorFlow/Keras) for binary classification.
* **Backend:** Flask API to serve the model predictions.
* **Frontend:** Conversational chatbot UI (HTML/JavaScript).
* **Explainability:** Uses the **SHAP** library to show the top 3 factors driving the risk score.

## 🚀 Deployment & Technology

| Component | Technology | Status |
| :--- | :--- | :--- |
| **Model** | TensorFlow / Keras | Ready |
| **Backend API** | Python / Flask | Ready |
| **Code Host** | GitHub | Live |
| **Cloud Host** | Render | *(Update URL after deployment)* |

## ⚙️ How to Run Locally

1.  **Clone the project:**
    ```bash
    git clone [https://github.com/akshbala-01/AI-Health-Chatbot.git](https://github.com/akshbala-01/AI-Health-Chatbot.git)
    cd AI-Health-Chatbot
    ```
2.  **Setup Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3.  **Run the Training Script** (to ensure all model files are fresh):
    ```bash
    python train_model.py
    ```
4.  **Start the Server:**
    ```bash
    python app.py
    ```
5.  **Access:** Open your browser to `http://127.0.0.1:5000/`.
