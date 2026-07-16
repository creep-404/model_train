# 🚗 Car Price Prediction using Machine Learning

A machine learning project that predicts the **new market price (EGP)** of cars using historical pricing and vehicle attributes. The project demonstrates a complete machine learning workflow, from data preprocessing and feature engineering to model training, evaluation, and prediction.

---

## 📌 Overview

This project uses supervised machine learning to estimate car prices from an Egyptian car marketplace dataset. Multiple regression models are trained and compared, with the best-performing model automatically selected and saved for future predictions.

---

## 📂 Project Structure

```text
model_train/
│
├── car_price.csv              # Dataset
├── train_model.py             # Model training pipeline
├── predict_example.py         # Example prediction script
├── car_price_model.joblib     # Trained model
├── model_evaluation.png       # Evaluation plots
├── model_results.json         # Model performance metrics
└── README.md
```

---

## ⚙️ Features

- Data cleaning and preprocessing
- Feature engineering
- Automatic categorical encoding
- Feature scaling
- Multiple regression model comparison
- Cross-validation
- Automatic best model selection
- Feature importance visualization
- Prediction using saved model

---

## 🤖 Machine Learning Models

The following regression algorithms are trained and evaluated:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The model with the highest performance is automatically selected and exported as a `.joblib` file.

---

## 📊 Features Used

The model learns from the following information:

- Vehicle Year
- Car Age at Listing
- Previous Price
- Listing Month
- Brand
- Transmission Type
- Trim Level
- Price Change Direction

---

## 📈 Evaluation Metrics

Model performance is evaluated using:

- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- R² Score
- 5-Fold Cross Validation

The project also generates:

- Predicted vs Actual Price Plot
- Top Feature Importance Chart

---

## ▶️ Training

Run the training script:

```bash
python train_model.py
```

The script will:

- Clean and preprocess the dataset
- Train multiple regression models
- Compare model performance
- Select the best-performing model
- Save the trained model
- Generate evaluation plots and performance metrics

---

## 💡 Making Predictions

Run:

```bash
python predict_example.py
```

Modify the input values inside the script to predict prices for different vehicles.

Example input:

```python
new_car = {
    "year": 2023,
    "car_age_at_listing": 1,
    "old_price": 700000,
    "listing_month": 6,
    "brand": "Toyota",
    "transmission": "Automatic",
    "trim_grouped": "Elgance",
    "change_direction": "up"
}
```

---

## 📦 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

---

## 📷 Output

After training, the project generates:

- Trained machine learning model (`car_price_model.joblib`)
- Model performance metrics (`model_results.json`)
- Predicted vs Actual visualization
- Feature Importance visualization

---

## 🚀 Future Improvements

- Hyperparameter optimization
- XGBoost, LightGBM, and CatBoost implementation
- Streamlit web application
- FastAPI deployment
- SHAP-based model explainability
- Automated model retraining pipeline

---

## 👨‍💻 Author

**Ahmad**

GitHub: **https://github.com/creep-404**