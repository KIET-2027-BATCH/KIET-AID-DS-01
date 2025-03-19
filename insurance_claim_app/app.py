from flask import Flask, render_template, request
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load("model/insurance_claim_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Dictionary to map regions to numeric values
region_map = {
    "northeast": 0,
    "northwest": 1,
    "southeast": 2,
    "southwest": 3
}

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def landing():
    return render_template('landing.html')  # Show landing page first

@app.route('/home')
def home():
    return render_template('index.html')  # Main index page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = float(request.form['age'])
        gender = request.form['gender']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']

        # Convert categorical inputs to numerical
        gender = 1 if gender == 'Male' else 0
        smoker = 1 if smoker == 'Yes' else 0
        region = region_map.get(region, 0)  # Default to 0 if not found

        # Prepare input data
        input_data = np.array([[age, gender, bmi, children, smoker, region]])

        # Scale input
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)

        # Convert result
        result = "**✅ CLAIM APPROVED**" if prediction[0] == 1 else "**❌ CLAIM REJECTED**\n\n" + "\n".join([
    "1️⃣ Incomplete or Incorrect Information",
    "2️⃣ Pre-existing Conditions not covered",
   
])



        return render_template('result.html', result=result)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
