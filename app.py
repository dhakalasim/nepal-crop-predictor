from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model
model = joblib.load('model.pkl')

# Read the HTML file
with open('index.html', 'r') as f:
    html_content = f.read()

@app.route('/')
def home():
    return html_content

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    new_data = {
        'temperature': [data['temperature']],
        'rainfall': [data['rainfall']],
        'humidity': [data['humidity']],
        'pH': [data['pH']],
        'N': [data['N']],
        'P': [data['P']],
        'K': [data['K']],
        'altitude': [data['altitude']]
    }
    new_df = pd.DataFrame(new_data)
    predicted_crop = model.predict(new_df)[0]
    return jsonify({'predicted_crop': predicted_crop})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
