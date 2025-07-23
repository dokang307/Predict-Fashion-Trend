# Predict-Fashion-Trend

This project predicts upcoming fashion trends using the **fashion_US_UK** dataset from Kaggle. It applies feature engineering, handles data imbalance, and leverages XGBoost for classification. The final result is visualized and will be deployed as a Streamlit web app.

## 📂 Dataset

- **Source**: [Fashion US UK Dataset - Kaggle](https://www.kaggle.com/)
- Contains fashion-related features including:
  - Brand, Category, Style Attributes, Color, Season
  - Mentions in Fashion Magazines & by Fashion Influencers
  - Customer Reviews, Purchase History, Social Media Comments

## 🧹 Data Processing

- Combined **Fashion Magazines** and **Fashion Influencers** into a total mention score
- Identified each product by:
  ```python
  product_features = ["Brand", "Category", "Style Attributes", "Color", "Season"] 
