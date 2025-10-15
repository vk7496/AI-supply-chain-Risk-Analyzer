# AI-supply-chain-Risk-Analyzer
Machine Learning dashboard that predicts and visualizes supplier risks in the supply chain using python ,streamlet AI analytics.Designed for manufacturing and food industry decision-makers.
# 📦 AI Supply Chain Risk Analyzer

### 🔹 Overview
This project uses **Artificial Intelligence and Machine Learning** to detect and visualize potential risks in the supply chain — such as supplier delays, rising costs, or delivery instability.

It is designed for manufacturing and food companies that need better visibility and control over their supplier performance.

---

### 🧠 Key Features
- Predicts supplier delay probability using ML (Random Forest)
- Displays interactive risk charts with **Plotly**
- Supports both sample and real supply chain CSV data
- 100% built with **Python + Streamlit**

---

### ⚙️ How It Works
1. Upload your CSV file containing supplier data (`Supplier`, `LeadTimeDays`, `CostIndex`, `OrderSize`, `Delay`).
2. The app trains a simple AI model to predict delay risk.
3. Interactive dashboard shows risk probabilities per supplier.

---

### 🧩 Tech Stack
- **Python 3.10+**
- **Streamlit**
- **Scikit-learn**
- **Plotly**
- **Pandas / NumPy**

---

### 📊 Sample Data
A demo dataset is included:  
[`supply_chain_sample.csv`](./supply_chain_sample.csv)

---

### 🚀 Run the App
```bash
streamlit run app.py
Future Improvements
Real - time supplier data integration via API
Advanced anomaly detection with LSTM models
Alert system for high-risk suppliers
Developed by [VISTA KAVIANI] -AI Developer
Oman golden bird LLc
