"""Inference and Decision Support System for Weather Prediction.

Loads the saved model pipeline to perform single or batch predictions,
outputting class probabilities and domain-specific decision support guidance
(Aviation, Agriculture, Logistics, Outdoor Events).
"""

import os
from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np
import joblib

from src.preprocess import clip_physical_outliers, engineer_features


class WeatherPredictor:
    """Production Inference Engine for Weather Type Classification."""

    def __init__(self, model_bundle_path: str = "models/weather_classifier_pipeline.joblib"):
        if not os.path.exists(model_bundle_path):
            raise FileNotFoundError(
                f"Model bundle not found at {model_bundle_path}. Please run train.py first."
            )
        self.bundle = joblib.load(model_bundle_path)
        self.model = self.bundle['model']
        self.model_name = self.bundle['model_name']
        self.preprocessor = self.bundle['preprocessor']
        self.label_encoder = self.bundle['label_encoder']
        self.class_names = self.bundle['class_names']
        self.num_cols = self.bundle['num_cols']
        self.cat_cols = self.bundle['cat_cols']

    def preprocess_input(self, input_df: pd.DataFrame) -> np.ndarray:
        """Apply outlier clipping, feature engineering, and standard transformation."""
        df_clean = clip_physical_outliers(input_df)
        df_eng = engineer_features(df_clean)
        return self.preprocessor.transform(df_eng)

    def predict(self, data: Union[Dict[str, Any], pd.DataFrame]) -> List[Dict[str, Any]]:
        """Predict weather category and class probabilities for input records."""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Input must be a dictionary or a pandas DataFrame.")

        X_proc = self.preprocess_input(df)
        pred_encoded = self.model.predict(X_proc)
        pred_labels = self.label_encoder.inverse_transform(pred_encoded)

        has_proba = hasattr(self.model, "predict_proba")
        probas = self.model.predict_proba(X_proc) if has_proba else None

        results = []
        for i in range(len(df)):
            record_proba = {}
            if probas is not None:
                for cls_idx, cls_name in enumerate(self.class_names):
                    record_proba[cls_name] = round(float(probas[i][cls_idx]), 4)

            results.append({
                'predicted_weather': pred_labels[i],
                'confidence': max(record_proba.values()) if record_proba else None,
                'class_probabilities': record_proba,
                'input_data': df.iloc[i].to_dict()
            })

        return results

    def predict_with_decision_support(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prediction combined with actionable operational recommendations."""
        pred_res = self.predict(input_data)[0]
        weather = pred_res['predicted_weather']
        conf = pred_res['confidence']
        probs = pred_res['class_probabilities']
        
        # Operational Decision Support Rules
        recommendations = {
            'Aviation & Flight Safety': [],
            'Agriculture & Irrigation': [],
            'Logistics & Ground Transport': [],
            'Outdoor Events & Public Advisory': []
        }
        
        temp = input_data.get('Temperature', 20)
        wind = input_data.get('Wind Speed', 10)
        vis = input_data.get('Visibility (km)', 10)
        precip = input_data.get('Precipitation (%)', 0)
        uv = input_data.get('UV Index', 3)
        
        # 1. Aviation
        if vis < 3.0:
            recommendations['Aviation & Flight Safety'].append("CRITICAL: Low visibility hazard (<3km). Initiate Instrument Flight Rules (IFR).")
        if wind > 30.0:
            recommendations['Aviation & Flight Safety'].append("WARNING: High wind speeds detected (>30 km/h). Crosswind landing precautions advised.")
        if weather == 'Snowy' or (temp <= 0 and precip > 40):
            recommendations['Aviation & Flight Safety'].append("ALERT: High risk of airframe icing and runway snow accumulation. De-icing required.")
        if not recommendations['Aviation & Flight Safety']:
            recommendations['Aviation & Flight Safety'].append("NORMAL: Atmospheric conditions optimal for standard VFR flight operations.")

        # 2. Agriculture
        if weather == 'Rainy' or precip > 60:
            recommendations['Agriculture & Irrigation'].append("RECOMMENDED: Postpone artificial irrigation; adequate precipitation expected.")
        elif weather == 'Sunny' and temp > 30:
            recommendations['Agriculture & Irrigation'].append("RECOMMENDED: Increase soil moisture replenishment; high evaporation rates.")
        elif weather == 'Snowy' or temp < 0:
            recommendations['Agriculture & Irrigation'].append("FROST ADVISORY: Deploy crop frost protection covers; soil freezing risk.")
        else:
            recommendations['Agriculture & Irrigation'].append("ROUTINE: Maintain scheduled soil monitoring and baseline crop care.")

        # 3. Logistics & Ground Transport
        if weather in ['Snowy', 'Rainy'] or vis < 4.0:
            recommendations['Logistics & Ground Transport'].append("ADVISORY: Wet or icy road surfaces expected. Enforce 20% speed reductions and increased vehicle following distance.")
        if wind > 35.0:
            recommendations['Logistics & Ground Transport'].append("WARNING: High winds pose tipping danger for high-profile transit vehicles.")
        if not recommendations['Logistics & Ground Transport']:
            recommendations['Logistics & Ground Transport'].append("CLEAR: Roadways dry, visibility standard. On-time delivery targets achievable.")

        # 4. Outdoor Events
        if weather == 'Rainy':
            recommendations['Outdoor Events & Public Advisory'].append("HIGH RISK: Rainfall anticipated. Relocate outdoor festivities to sheltered venues.")
        elif weather == 'Sunny' and uv >= 7:
            recommendations['Outdoor Events & Public Advisory'].append("HEAT & UV WARNING: UV Index is extreme (>=7). Provide shaded hydration zones and sun protection.")
        elif weather == 'Snowy':
            recommendations['Outdoor Events & Public Advisory'].append("WINTER ADVISORY: Cold stress hazard. Restrict extended outdoor exposure.")
        else:
            recommendations['Outdoor Events & Public Advisory'].append("FAVORABLE: Mild, pleasant conditions for public gatherings and sporting events.")

        return {
            'prediction_summary': {
                'model_used': self.model_name,
                'predicted_weather': weather,
                'confidence_score': f"{conf * 100:.1f}%",
                'probabilities': probs
            },
            'decision_support': recommendations
        }


def run_demo_predictions() -> None:
    """Test scenario predictions on representative meteorological profiles."""
    predictor = WeatherPredictor()
    
    scenarios = [
        {
            'name': 'Cold Winter Mountain Blizzard',
            'data': {
                'Temperature': -8.0,
                'Humidity': 88,
                'Wind Speed': 24.5,
                'Precipitation (%)': 85.0,
                'Cloud Cover': 'overcast',
                'Atmospheric Pressure': 995.0,
                'UV Index': 1,
                'Season': 'Winter',
                'Visibility (km)': 1.8,
                'Location': 'mountain'
            }
        },
        {
            'name': 'Hot Summer Coastal Beach Day',
            'data': {
                'Temperature': 36.5,
                'Humidity': 42,
                'Wind Speed': 7.0,
                'Precipitation (%)': 5.0,
                'Cloud Cover': 'clear',
                'Atmospheric Pressure': 1014.5,
                'UV Index': 11,
                'Season': 'Summer',
                'Visibility (km)': 10.0,
                'Location': 'coastal'
            }
        },
        {
            'name': 'Monsoon Inland Thunderstorm',
            'data': {
                'Temperature': 18.0,
                'Humidity': 95,
                'Wind Speed': 18.0,
                'Precipitation (%)': 92.0,
                'Cloud Cover': 'overcast',
                'Atmospheric Pressure': 990.2,
                'UV Index': 2,
                'Season': 'Autumn',
                'Visibility (km)': 2.5,
                'Location': 'inland'
            }
        }
    ]

    print("\n" + "=" * 70)
    print("DEMONSTRATION: WEATHER PREDICTION & DECISION SUPPORT SYSTEM")
    print("=" * 70)

    for scenario in scenarios:
        print(f"\n[SCENARIO]: {scenario['name']}")
        result = predictor.predict_with_decision_support(scenario['data'])
        summary = result['prediction_summary']
        
        print(f"  Predicted Category: >> {summary['predicted_weather'].upper()} << (Confidence: {summary['confidence_score']})")
        print(f"  Class Probabilities: {summary['probabilities']}")
        print("  Actionable Operational Guidance:")
        for domain, advice in result['decision_support'].items():
            for item in advice:
                print(f"    - [{domain}]: {item}")


if __name__ == "__main__":
    run_demo_predictions()
