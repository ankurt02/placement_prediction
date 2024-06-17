from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the model
model = pickle.load(open('C:\\Users\\acer\\Desktop\\all\\Flutter\\placement_prediction\\flask_app\\flask_app_file\\model.pkl', 'rb'))

@app.route('/')
def home():
    return "API is running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    try:
        CGPA = float(data['CGPA'])
        Internships = float(data['Internships'])
        Projects = float(data['Projects'])
        AptitudeTestScore = float(data['AptitudeTestScore'])
        SoftSkillsRating = float(data['SoftSkillsRating'])
        ExtracurricularActivities = float(data['ExtracurricularActivities'])
        PlacementTraining = float(data['PlacementTraining'])
        SSC_Marks = float(data['SSC_Marks'])
        HSC_Marks = float(data['HSC_Marks'])

        arr = np.array([CGPA, Internships, Projects, AptitudeTestScore, SoftSkillsRating, ExtracurricularActivities, PlacementTraining, SSC_Marks, HSC_Marks])
        prediction = model.predict([arr])
        output = int(prediction[0])
        print(output)
    except Exception as e:
        return jsonify({'error': str(e)})

    return jsonify({'placement': output})

if __name__ == "__main__":
    model = pickle.load(open('model.pkl', 'rb'))

# Define sample inputs
    sample_inputs = [
    [8.5, 2, 3, 90, 8, 5, 7, 85, 80],
    [6.0, 1, 1, 70, 6, 3, 5, 75, 70],
    [9.0, 3, 4, 95, 9, 7, 8, 90, 85]
]

    for inputs in sample_inputs:
        arr = np.array(inputs)
        prediction = model.predict([arr])
        print(f'Input: {inputs} => Prediction: {prediction}')

    app.run(debug=True)