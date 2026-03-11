import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AI Energy Predictor", layout="wide")

# ANN architecture
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4,6),
            nn.ReLU(),
            nn.Linear(6,6),
            nn.ReLU(),
            nn.Linear(6,1)
        )

    def forward(self,x):
        return self.model(x)

# load model
model = ANN()
model.load_state_dict(torch.load("best_model.pt", map_location=torch.device("cpu")))
model.eval()

# load scaler
scaler = pickle.load(open("scaler.pkl","rb"))

# load dataset
df = pd.read_csv("powerplant.csv")

# header
st.title("⚡ AI Energy Intelligence Dashboard")
st.markdown("Deep Learning powered prediction system for power plant output")

# sidebar controls
st.sidebar.header("Input Controls")

AT = st.sidebar.slider("Temperature", float(df.AT.min()), float(df.AT.max()), 15.0)
V = st.sidebar.slider("Vacuum", float(df.V.min()), float(df.V.max()), 40.0)
AP = st.sidebar.slider("Pressure", float(df.AP.min()), float(df.AP.max()), 1010.0)
RH = st.sidebar.slider("Humidity", float(df.RH.min()), float(df.RH.max()), 80.0)

predict = st.sidebar.button("Run AI Prediction")

# layout
col1, col2, col3 = st.columns(3)

if predict:

    data = np.array([[AT,V,AP,RH]])
    data_scaled = scaler.transform(data)

    tensor = torch.tensor(data_scaled).float()

    with torch.no_grad():
        prediction = model(tensor).numpy()[0][0]

    col1.metric("Predicted Output", f"{prediction:.2f}")
    col2.metric("Dataset Size", len(df))
    col3.metric("Features Used", 4)

# charts section
st.markdown("---")

colA, colB = st.columns(2)

with colA:

    st.subheader("Power Output Distribution")

    fig = px.histogram(df, x="PE", nbins=40)

    st.plotly_chart(fig, use_container_width=True)

with colB:

    st.subheader("Feature Correlation")

    corr = df.corr()

    fig2 = px.imshow(corr, text_auto=True, aspect="auto")

    st.plotly_chart(fig2, use_container_width=True)

# scatter analysis
st.subheader("Temperature vs Power Output")

fig3 = px.scatter(df, x="AT", y="PE", color="RH")

st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.markdown("Built with PyTorch + Streamlit | AI Model by Arunesh Singh Rajawat")