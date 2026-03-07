import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Sample data (expand with real Nepal agriculture data for better accuracy)
data = {
    'temperature': [25, 20, 28, 15, 22, 18, 30, 12, 24, 16],
    'rainfall': [150, 100, 200, 80, 120, 90, 180, 70, 140, 110],
    'humidity': [80, 70, 85, 65, 75, 68, 82, 60, 78, 72],
    'pH': [6.5, 7.0, 6.0, 7.5, 6.8, 7.2, 5.8, 7.8, 6.2, 7.1],
    'N': [50, 40, 60, 30, 45, 35, 55, 25, 52, 42],
    'P': [30, 25, 35, 20, 28, 22, 32, 18, 31, 26],
    'K': [40, 35, 45, 30, 38, 32, 42, 28, 41, 36],
    'altitude': [100, 2000, 500, 3000, 1500, 2500, 300, 3500, 800, 2200],
    'crop': ['rice', 'wheat', 'maize', 'potato', 'millet', 'wheat', 'rice', 'potato', 'maize', 'millet']
}

df = pd.DataFrame(data)
X = df.drop('crop', axis=1)
y = df['crop']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'model.pkl')
print("Model trained and saved as model.pkl")
