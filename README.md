# 🔧 PrognisAI – Predictive Maintenance using Deep Learning

An AI-powered predictive maintenance system that estimates the **Remaining Useful Life (RUL)** of engines using deep learning models (**LSTM & GRU**) and presents insights through an **interactive dashboard**.

---

## 🚀 Project Overview

Predictive maintenance helps industries avoid unexpected failures by forecasting when equipment is likely to fail.

This project uses the **NASA CMAPSS dataset** to:

* Predict engine RUL
* Detect potential failures
* Generate intelligent alerts
* Visualize results through a modern dashboard

---

## 🧠 Features

* 🔹 Deep Learning Models (LSTM & GRU)
* 🔹 Ensemble Model (Weighted combination)
* 🔹 RMSE-based model comparison
* 🔹 Dynamic alert system (Healthy / Warning / Critical)
* 🔹 Interactive dashboard (Streamlit)
* 🔹 Real-time engine inspection
* 🔹 Advanced visualizations (Plotly charts + Gauge meter)

---

## 📊 Tech Stack

**Machine Learning:**

* Python
* TensorFlow / Keras
* Scikit-learn
* NumPy / Pandas

**Visualization & UI:**

* Streamlit
* Plotly
* Matplotlib

---

## 📁 Project Structure

```
PROGNISAI/
│
├── app.py                  # Streamlit Dashboard
├── main.ipynb             # Model training & evaluation
│
├── models/
│   ├── lstm_rul_model.h5
│   └── gru_rul_model.h5
│
├── data/
│   ├── train_FD001.txt ...
│   ├── test_FD001.txt ...
│   ├── RUL_FD001.txt ...
│
├── outputs/
│   ├── alert_data.csv
│   └── model_results.csv
│
├── dataset_load.py
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/PrognisAI.git
cd PrognisAI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit pandas numpy matplotlib plotly tensorflow scikit-learn
```

---

## ▶️ Running the Project

### 1️⃣ Train the Model (Optional)

Open and run:

```bash
main.ipynb
```

This will:

* Train LSTM & GRU models
* Generate predictions
* Save results in `/outputs`

---

### 2️⃣ Run Dashboard

```bash
python -m streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 📈 Dashboard Features

* 📊 Model performance comparison (RMSE)
* 📈 RUL prediction trends
* 🚨 Alert distribution (Healthy / Warning / Critical)
* 🎯 Engine-level inspection
* ⚙️ Filtering system
* 🔥 RUL gauge visualization
* 📉 Error analysis

---

## 🧪 Models Used

| Model    | Description                         |
| -------- | ----------------------------------- |
| LSTM     | Captures long-term dependencies     |
| GRU      | Faster and efficient sequence model |
| Ensemble | Weighted combination (LSTM + GRU)   |

---

## 📊 Evaluation Metrics

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* R² Score

---

## 🚨 Alert System

| Status      | Condition       |
| ----------- | --------------- |
| 🟢 Healthy  | RUL > 0.6       |
| 🟡 Warning  | 0.3 < RUL ≤ 0.6 |
| 🔴 Critical | RUL ≤ 0.3       |

---

## 📌 Key Insights

* GRU model performs slightly better than LSTM
* Ensemble improves generalization
* Dashboard enables real-time monitoring of engine health
* Critical engines can be identified instantly

---

## 🎯 Future Improvements

* 🌐 Deploy dashboard online
* 📡 Real-time data streaming
* 📂 Upload new engine data for prediction
* 🤖 Auto-retraining pipeline
* 📊 Advanced anomaly detection

---

## 👨‍💻 Author

**Cahal Agarwalla**

---

## ⭐ Acknowledgements

* NASA CMAPSS Dataset
* TensorFlow / Keras
* Streamlit & Plotly


