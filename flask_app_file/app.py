from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return "API is running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
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

    return jsonify({'placement': output})

if __name__ == "__main__":
    app.run(debug=True)
