import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Theory",
    layout="centered"
)

st.title("Theory")

st.sidebar.header("Navigation")
points = st.sidebar.slider("Number of Points", 10, 500, 100)
seed = st.sidebar.number_input("Seed", min_value=0, max_value=1000, value=42)
generate = st.sidebar.button("Generate Data")

if generate:
    np.random.seed(seed)
    x = np.linspace(0, 10, points)
    y = np.sin(x) + np.random.normal(0, 0.1, points)
    
    data = pd.DataFrame({'x': x, 'y': y})
    
    st.success("Data generated successfully!")
    
    st.subheader("Generated Data")
    st.write(data)

    st.subheader("Plot")
    st.line_chart(data.set_index('x'))
    