# crop_predictor.py
# A simple AI model to predict best crops for Nepal's diverse weather (e.g., mountains/valleys).
# Uses Random Forest to recommend crops based on temperature, rainfall, humidity, pH, soil nutrients, and altitude.
# Sample data included; replace with real data for better accuracy.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Create sample data (features: weather/soil, labels: crops)
# Features: temperature (C), rainfall (mm), humidity (%), pH, N (kg/ha), P (kg/ha), K (kg/ha), altitude (m)
# Crops: rice, wheat, maize, potato, millet (common in Nepal)
data = {
    'temperature': [25, 20, 28, 15, 22, 18, 30, 12, 24, 16],
    'rainfall': [150, 100, 200, 80, 120, 90, 180, 70, 140, 110],
    'humidity': [80, 70, 85, 65, 75, 68, 82, 60, 78, 72],
    'pH': [6.5, 7.0, 6.0, 7.5, 6.8, 7.2, 5.8, 7.8, 6.2, 7.1],
    'N': [50, 40, 60, 30, 45, 35, 55, 25, 52, 42],
    'P': [30, 25, 35, 20, 28, 22, 32, 18, 31, 26],
    'K': [40, 35, 45, 30, 38, 32, 42, 28, 41, 36],
    'altitude': [100, 2000, 500, 3000, 1500, 2500, 300, 3500, 800, 2200],  # Simulates valleys (low) to mountains (high)
    'crop': ['rice', 'wheat', 'maize', 'potato', 'millet', 'wheat', 'rice', 'potato', 'maize', 'millet']
}

df = pd.DataFrame(data)

# Step 2: Prepare data for training
X = df.drop('crop', axis=1)  # Features
y = df['crop']  # Labels

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the model (Random Forest - like a smart computer brain guessing based on patterns)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Test the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Step 5: Predict for new data (example: mountain area with cool, rainy weather)
new_data = {
    'temperature': [18],
    'rainfall': [120],
    'humidity': [75],
    'pH': [6.5],
    'N': [45],
    'P': [28],
    'K': [38],
    'altitude': [2500]  # High altitude like Nepal mountains
}

new_df = pd.DataFrame(new_data)
predicted_crop = model.predict(new_df)
print(f"Best crop for this weather/soil: {predicted_crop[0]}")

# To improve: Add more data, save model with joblib, or integrate weather API for real-time input.
