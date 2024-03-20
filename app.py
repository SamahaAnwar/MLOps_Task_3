from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
#from Heart_Disease_Prediction import model
import pickle

app = Flask(__name__)

# Define a route for the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json
    
    # Preprocess input data if necessary
    preprocessed_data = preprocess(data)
    
    # Use your machine learning model to make predictions
    model = pickle.load(open("model.pkl", "rb"))
    prediction = model.predict(preprocessed_data)
    
    # Return the prediction as JSON
    return jsonify({'prediction': prediction})

# Function to preprocess input data
def preprocess(data):
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    return X

if __name__ == '__main__':
    # Run the Flask app
    app.run(host ="0.0.0.0", debug=True)