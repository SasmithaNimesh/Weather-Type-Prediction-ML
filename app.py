"""Interactive Web Application for Weather Type Prediction & Decision Support.

Flask-based web dashboard providing real-time inference, interactive sliders,
visual probability breakdowns, and operational decision-support recommendations.
"""

import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from src.predict import WeatherPredictor

app = Flask(__name__)
predictor = WeatherPredictor("models/weather_classifier_pipeline.joblib")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/showcase')
def showcase():
    with open('project_showcase.html', 'r', encoding='utf-8') as f:
        return f.read()


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        data = request.get_json(force=True)
        # Parse and type-cast
        features = {
            'Temperature': float(data.get('Temperature', 20.0)),
            'Humidity': float(data.get('Humidity', 60.0)),
            'Wind Speed': float(data.get('Wind Speed', 10.0)),
            'Precipitation (%)': float(data.get('Precipitation (%)', 20.0)),
            'Cloud Cover': str(data.get('Cloud Cover', 'partly cloudy')),
            'Atmospheric Pressure': float(data.get('Atmospheric Pressure', 1013.0)),
            'UV Index': int(data.get('UV Index', 4)),
            'Season': str(data.get('Season', 'Spring')),
            'Visibility (km)': float(data.get('Visibility (km)', 8.0)),
            'Location': str(data.get('Location', 'inland'))
        }
        
        result = predictor.predict_with_decision_support(features)
        return jsonify({'status': 'success', 'data': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/reports/figures/<path:filename>')
def serve_figure(filename):
    return send_from_directory('reports/figures', filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Weather App running at: http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
