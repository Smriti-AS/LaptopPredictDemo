from flask import Flask, request, render_template
import pickle
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)

print("Starting Flask app...")

# Verify model files
print("Checking model files...")
print(f"RF Model exists: {os.path.exists('rf_model.pkl')}")
print(f"Scaler exists: {os.path.exists('min_max_scaler_X.pkl')}")
print(f"Dummy columns exist: {os.path.exists('dummy_columns.pkl')}")

# Load the trained Random Forest model
try:
    with open('rf_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("Error: RF model file not found.")
    exit(1)

# Load the scaler
try:
    with open('min_max_scaler_X.pkl', 'rb') as file:
        min_max_scaler_X = pickle.load(file)
except FileNotFoundError:
    print("Error: Scaler file not found.")
    exit(1)
    
# Load the target scaler
try:
    with open('models/min_max_scaler_y.pkl', 'rb') as file:
        min_max_scaler_y = pickle.load(file)
except FileNotFoundError:
    print("Error: Target scaler file not found.")
    exit(1)


# Load dummy columns
try:
    with open('dummy_columns.pkl', 'rb') as file:
        dummy_columns = pickle.load(file)
except FileNotFoundError:
    print("Error: Dummy columns file not found.")
    exit(1)

print("Model and preprocessors loaded successfully.")


def preprocess_input(data):
    """
    Preprocesses the input data to align it with the training data format.
    """
    # Convert input dictionary to DataFrame
    df = pd.DataFrame([data])

    # Apply one-hot encoding
    df = pd.get_dummies(df)

    # Align columns with the training dummy columns
    df = df.reindex(columns=dummy_columns, fill_value=0)

    # Debugging: Log column alignment
    print(f"Columns in input data after reindexing:\n{df.columns}")
    print(f"Dummy columns expected by model:\n{dummy_columns}")

    # Scale the data
    scaled_data = min_max_scaler_X.transform(df)
    return scaled_data


@app.route('/')
def home():
    """
    Renders the homepage with the prediction form.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles form submission, processes input, and predicts the output.
    """
    try:
        # Extract form inputs
        input_data = {
            "Ram": int(request.form.get('Ram', 0)),
            "Weight": float(request.form.get('Weight', 0.0)),
            "Touchscreen": int(request.form.get('Touchscreen', 0)),
            "IPS": int(request.form.get('IPS', 0)),
            "PPI": int(request.form.get('PPI', 0)),
            "Cpu brand": request.form.get('Cpu brand', 'Unknown'),
            "HDD": int(request.form.get('HDD', 0)),
            "SSD": int(request.form.get('SSD', 0)),
            "Hybrid": int(request.form.get('Hybrid', 0)),
            "Flash_Storage": int(request.form.get('Flash_Storage', 0)),
            "Gpu brand": request.form.get('Gpu brand', 'Unknown'),
            "os": request.form.get('os', 'Unknown')
        }

        # Debugging: Log raw input
        print(f"Raw input: {input_data}")

        # Preprocess input
        preprocessed_input = preprocess_input(input_data)

        # Debugging: Verify alignment
        print(f"Preprocessed input shape: {preprocessed_input.shape}")
        print(f"Model expects: {model.n_features_in_} features")

        # Make scaled prediction
        prediction_scaled = model.predict(preprocessed_input)[0]

        # Rescale prediction back to original range
        prediction_unscaled = min_max_scaler_y.inverse_transform([[prediction_scaled]])[0][0]

        return render_template('index.html', prediction=round(prediction_unscaled, 2), error=None)

    except Exception as e:
        # Debugging: Print the exception in the console
        print(f"Error during prediction: {e}")
        return render_template('index.html', prediction=None, error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
