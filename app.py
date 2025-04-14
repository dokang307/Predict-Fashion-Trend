import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import chardet

# Load trained model
with open(r"D:\Khang\FPT semester\semester_4\DAP391m\final_project\.venv\final_attempt\fashion_trend_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load encoders
with open(r"D:\Khang\FPT semester\semester_4\DAP391m\final_project\.venv\final_attempt\encoders.pkl", "rb") as file:
    encoders = pickle.load(file)

def read_file(uploaded_file):
    # Kiểm tra định dạng file
    if uploaded_file.name.endswith('.csv'):
        # Tự động phát hiện encoding
        raw_data = uploaded_file.read()
        detected_encoding = chardet.detect(raw_data)['encoding']
        
        # Đọc file với encoding phát hiện được
        uploaded_file.seek(0)  # Reset file pointer
        df = pd.read_csv(uploaded_file, encoding=detected_encoding, low_memory=False)
    
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
    
    return df

# Function to preprocess input data
def preprocess_input(data):
    categorical_features = ['Brand', 'Category', 'Style Attributes', 'Color', 'Season']
    for col in categorical_features:
        if col in encoders:
            data[col] = encoders[col].transform(data[col])
    return data

def inverse_transform(df):
    """Chuyển các giá trị số về dạng text dựa trên encoders"""
    df_decoded = df.copy()
    for col, encoder in encoders.items():
        if col in df_decoded.columns:
            df_decoded[col] = encoder.inverse_transform(df_decoded[col])
    return df_decoded

# Function for data visualization
def visualize_data(df):
    st.subheader("Data Overview")
    st.write(df.head())
    
    st.subheader("Feature Distributions")
    selected_feature = st.selectbox("Select a feature to visualize", df.columns)
    fig, ax = plt.subplots()
    sns.histplot(df[selected_feature], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

# Function to make predictions
def predict_fashion_trend(input_data):
    processed_data = preprocess_input(pd.DataFrame([input_data]))
    prediction = model.predict(processed_data)
    return "Trend" if prediction[0] == 1 else "Not Trend"

# Function to handle CSV uploads and batch predictions
def batch_predict(uploaded_file):
    df = read_file(uploaded_file)
    processed_df = preprocess_input(df)
    predictions = model.predict(processed_df)
    df['Prediction'] = ["Trend" if p == 1 else "Not Trend" for p in predictions]
    # Chuyển từ dạng số về dạng chữ
    df_final = inverse_transform(df)
    return df_final


# Streamlit UI
st.title("Fashion Trend Prediction App")
option = st.sidebar.selectbox("Select Feature", ["Predict", "Analyze & Visualize Data", "Batch Prediction"])

if option == "Predict":
    st.subheader("Predict Fashion Trend")
    price = st.number_input("Price", min_value=0.0, step=1.0)
    brand = st.selectbox("Brand", encoders['Brand'].classes_)
    category = st.selectbox("Category", encoders['Category'].classes_)
    style_attributes = st.selectbox("Style Attributes", encoders['Style Attributes'].classes_)
    color = st.selectbox("Color", encoders['Color'].classes_)
    season = st.selectbox("Season", encoders['Season'].classes_)
    
    input_data = {
        "Price": price,
        "Brand": brand,
        "Category": category,
        "Style Attributes": style_attributes,
        "Color": color,
        "Season": season
    }
    
    if st.button("Predict"):
        result = predict_fashion_trend(input_data)
        st.write(f"Prediction: {result}")

elif option == "Analyze & Visualize Data":
    st.subheader("Data Analysis and Visualization")
    uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv", "xls", "xlsx"])
    if uploaded_file is not None:
        df = read_file(uploaded_file)
        visualize_data(df)

elif option == "Batch Prediction":
    st.subheader("Upload a CSV File for Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv", "xls", "xlsx"])
    if uploaded_file is not None:
        result_df = batch_predict(uploaded_file)
        st.write(result_df.head())
        st.download_button("Download Predictions", result_df.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
