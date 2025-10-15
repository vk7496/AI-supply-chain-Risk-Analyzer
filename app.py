

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px

# --- Title ---
st.title("📦 AI Supply Chain Risk Analyzer")
st.markdown("""
This dashboard predicts and visualizes **supply chain risk** — such as delivery delays or supplier instability — using machine learning.
""")

# --- Upload CSV ---
st.sidebar.header("Upload Your Supply Chain Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# --- Example dataset if none uploaded ---
if uploaded_file is None:
    st.info("No file uploaded. Using sample dataset.")
    np.random.seed(42)
    data = pd.DataFrame({
        'Supplier': np.random.choice(['A', 'B', 'C', 'D', 'E'], 100),
        'LeadTimeDays': np.random.randint(2, 15, 100),
        'CostIndex': np.random.uniform(0.8, 1.3, 100),
        'OrderSize': np.random.randint(50, 500, 100),
        'Delay': np.random.choice([0, 1], 100, p=[0.7, 0.3])
    })
else:
    data = pd.read_csv(uploaded_file)

# --- Display data ---
st.subheader("📊 Uploaded / Sample Data")
st.write(data.head())

# --- ML Model ---
X = data[['LeadTimeDays', 'CostIndex', 'OrderSize']]
y = data['Delay']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# --- Results ---
st.subheader("🧠 Model Accuracy")
st.metric(label="Prediction Accuracy", value=f"{accuracy * 100:.2f}%")

# --- Risk Visualization ---
data['PredictedRisk'] = model.predict(X)
risk_by_supplier = data.groupby('Supplier')['PredictedRisk'].mean().reset_index()

fig = px.bar(risk_by_supplier, x='Supplier', y='PredictedRisk',
             title="📉 Predicted Delay Risk by Supplier",
             labels={'PredictedRisk': 'Risk Probability (0–1)'},
             text_auto='.2f')
st.plotly_chart(fig, use_container_width=True)

st.success("✅ Analysis complete. Use this insight to mitigate supplier risk and optimize inventory planning.")
