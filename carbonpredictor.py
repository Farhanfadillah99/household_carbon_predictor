import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
import os


def log(msg):
    print(f" {msg}")


def train_model(data_path='data_energy.csv'):
    df = pd.read_csv(data_path)
    
    X = df[['electricity_kwh', 'gas_m3', 'transport_km']]
    y = df['carbon_kg']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    log("Training model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate R² score
    r2 = r2_score(y_test, y_pred)
    
    
    log(f"Model trained with R² score: {r2:.4f}")
    
    log("Saving model...")
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/carbon_predictor.pkl')


def predict(electricity_kwh, gas_m3, transport_km):
    model_path = 'model/carbon_predictor.pkl'
    if not os.path.exists(model_path):
        train_model()
    
    model = joblib.load(model_path)
    input_data = np.array([[electricity_kwh, gas_m3, transport_km]])
    prediction = model.predict(input_data)
    
    log(f"Estimated carbon emission: {prediction[0]:.2f} kg CO2")
    return prediction[0]


if __name__ == "__main__":
    log("Welcome to AI Green Tech!")
    
    electricity = int(input("Konsumsi Listrik (kWh/bulan) anda: "))
    gas = int(input("Konsumsi Gas (m3/bulan) anda: "))
    transport = int(input("Jarak Transportasi (km/bulan) anda: "))
    
    predict(electricity, gas, transport)

