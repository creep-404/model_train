"""
Example: load the trained model and predict a new car's price.
Run: python3 predict_example.py
"""
import joblib
import pandas as pd

model = joblib.load("car_price_model.joblib")

# Fill in details for the car you want to price
new_car = pd.DataFrame([{
    "year": 2023,
    "car_age_at_listing": 1,          # listing_year - year
    "old_price": 700000,              # last known price (EGP)
    "listing_month": 6,
    "brand": "Toyota",
    "transmission": "Automatic",
    "trim_grouped": "Elgance",        # use "Other" if unsure / not a common trim
    "change_direction": "up",         # up / down / flat
}])

predicted_price = model.predict(new_car)[0]
print(f"Predicted New Price: {predicted_price:,.0f} EGP")
