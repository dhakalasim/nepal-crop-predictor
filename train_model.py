import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

# Fetch real crop prediction dataset from online source
print("Fetching real agricultural data...")
try:
    # Try multiple sources
    urls = [
        'https://raw.githubusercontent.com/nishantsingh/Crop-Recommendation-System/main/data/Crop_recommendation.csv',
        'https://kaggle.com/api/v1/datasets/download/atharvaverma/crop-recommendation-system',  # This may fail without auth
    ]
    
    df = None
    for url in urls:
        try:
            df = pd.read_csv(url)
            print(f"Successfully loaded real dataset from {url}")
            print(f"Dataset has {len(df)} records with columns: {df.columns.tolist()}")
            break
        except:
            continue
    
    # If no online source worked, create expanded realistic data
    if df is None:
        print("Creating expanded realistic agricultural dataset...")
        # Generate realistic crop data based on agricultural science
        import numpy as np
        np.random.seed(42)
        
        n_samples = 500
        crops_list = ['rice', 'wheat', 'maize', 'potato', 'millet', 'barley', 'sugarcane', 'cotton']
        
        # Define typical ranges for each crop
        crop_ranges = {
            'rice': {'temp': (20, 30), 'rain': (150, 250), 'hum': (70, 90), 'pH': (6.0, 7.0), 'N': (80, 120), 'P': (40, 60), 'K': (40, 60), 'alt': (50, 500)},
            'wheat': {'temp': (15, 25), 'rain': (80, 150), 'hum': (40, 60), 'pH': (6.5, 7.5), 'N': (60, 100), 'P': (30, 50), 'K': (30, 50), 'alt': (200, 1500)},
            'maize': {'temp': (18, 27), 'rain': (100, 200), 'hum': (60, 80), 'pH': (6.0, 7.0), 'N': (100, 150), 'P': (40, 60), 'K': (40, 60), 'alt': (100, 1200)},
            'potato': {'temp': (15, 24), 'rain': (120, 200), 'hum': (70, 90), 'pH': (5.5, 7.0), 'N': (100, 150), 'P': (50, 80), 'K': (100, 150), 'alt': (500, 2500)},
            'millet': {'temp': (25, 35), 'rain': (50, 100), 'hum': (30, 50), 'pH': (6.0, 8.0), 'N': (30, 60), 'P': (20, 40), 'K': (20, 40), 'alt': (100, 1000)},
            'barley': {'temp': (10, 20), 'rain': (80, 150), 'hum': (40, 60), 'pH': (6.5, 8.0), 'N': (50, 90), 'P': (30, 50), 'K': (30, 50), 'alt': (300, 1800)},
            'sugarcane': {'temp': (20, 30), 'rain': (150, 250), 'hum': (60, 80), 'pH': (6.0, 7.5), 'N': (120, 180), 'P': (60, 100), 'K': (100, 150), 'alt': (50, 1000)},
            'cotton': {'temp': (20, 30), 'rain': (50, 100), 'hum': (40, 60), 'pH': (6.5, 8.0), 'N': (80, 120), 'P': (40, 60), 'K': (40, 60), 'alt': (100, 800)},
        }
        
        data = {
            'temperature': [],
            'rainfall': [],
            'humidity': [],
            'pH': [],
            'N': [],
            'P': [],
            'K': [],
            'altitude': [],
            'crop': []
        }
        
        samples_per_crop = n_samples // len(crops_list)
        
        for crop in crops_list:
            ranges = crop_ranges[crop]
            for _ in range(samples_per_crop):
                data['temperature'].append(np.random.uniform(ranges['temp'][0], ranges['temp'][1]))
                data['rainfall'].append(np.random.uniform(ranges['rain'][0], ranges['rain'][1]))
                data['humidity'].append(np.random.uniform(ranges['hum'][0], ranges['hum'][1]))
                data['pH'].append(np.random.uniform(ranges['pH'][0], ranges['pH'][1]))
                data['N'].append(np.random.uniform(ranges['N'][0], ranges['N'][1]))
                data['P'].append(np.random.uniform(ranges['P'][0], ranges['P'][1]))
                data['K'].append(np.random.uniform(ranges['K'][0], ranges['K'][1]))
                data['altitude'].append(np.random.uniform(ranges['alt'][0], ranges['alt'][1]))
                data['crop'].append(crop)
        
        df = pd.DataFrame(data)
        print(f"Generated {len(df)} realistic agricultural samples")
    else:
        # Process the fetched data
        if 'label' in df.columns:
            df = df.rename(columns={'label': 'crop'})
        elif 'crop_name' in df.columns:
            df = df.rename(columns={'crop_name': 'crop'})
        
        # Handle column mapping
        column_mapping = {
            'Temparature': 'temperature',
            'temperature': 'temperature',
            'Rainfall': 'rainfall',
            'rainfall': 'rainfall',
            'Humidity': 'humidity',
            'humidity': 'humidity',
            'ph': 'pH',
            'Ph': 'pH',
            'pH': 'pH',
            'Nitrogen': 'N',
            'N': 'N',
            'Phosphorus': 'P',
            'P': 'P',
            'Potassium': 'K',
            'K': 'K'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Select only required features
        required_features = ['temperature', 'rainfall', 'humidity', 'pH', 'N', 'P', 'K']
        
        if 'altitude' not in df.columns:
            df['altitude'] = 1000
        
        required_features.append('altitude')
        required_features.append('crop')
        
        available_cols = [col for col in required_features if col in df.columns]
        df = df[available_cols]
        df = df.dropna()
    
    print(f"Data shape: {df.shape}")
    print(f"Unique crops: {df['crop'].nunique()}")
    print(f"Crop distribution:\n{df['crop'].value_counts()}")
    
except Exception as e:
    print(f"Error: {e}")
    print("Using fallback sample data...")

# Prepare data for training
print("\nPreparing data for model training...")
X = df.drop('crop', axis=1)
y = df['crop']

print(f"Training set size: {len(X)}")
print(f"Features: {X.columns.tolist()}")

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model accuracy on test set: {accuracy * 100:.2f}%")

# Feature importance
print("\nFeature importance:")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"  {feature}: {importance:.4f}")

# Save the trained model
joblib.dump(model, 'model.pkl')
print("\nModel trained and saved as model.pkl")
