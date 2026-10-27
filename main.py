"""Main Orchestration Script for Weather Type Prediction Project.

Runs the complete lifecycle:
1. Exploratory Data Analysis & Visualization Generation
2. Preprocessing, Feature Engineering & Mutual Information Scoring
3. Multimodel Benchmark, 5-Fold Cross-Validation & Hyperparameter Tuning
4. Final Model Evaluation & Production Bundle Serialization
5. Practical Decision Support Simulation
"""

import sys
import os

from src.eda import run_eda
from src.train import run_training_pipeline
from src.predict import run_demo_predictions


def main():
    print("=" * 80)
    print("  WEATHER TYPE PREDICTION USING MACHINE LEARNING PIPELINE")
    print("  Author: Data Analysis | Machine Learning | Predictive Analytics")
    print("=" * 80)
    
    data_path = "data/weather_classification_data.csv"
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at '{data_path}'.")
        print("Please ensure kagglehub has downloaded the dataset.")
        sys.exit(1)
        
    # Step 1: Run EDA
    print("\n[STEP 1/3] Executing Exploratory Data Analysis...")
    run_eda(data_path=data_path, output_dir="reports")
    
    # Step 2: Run Training & Evaluation Pipeline
    print("\n[STEP 2/3] Training, Tuning, and Evaluating ML Models...")
    run_training_pipeline(data_path=data_path)
    
    # Step 3: Run Inference & Decision-Support Demo
    print("\n[STEP 3/3] Running Decision-Support Inference Scenarios...")
    run_demo_predictions()
    
    print("\n" + "=" * 80)
    print("SUCCESS: ALL PIPELINE STAGES COMPLETED!")
    print("Review artifacts in:")
    print(" - Visualizations: reports/figures/")
    print(" - Metrics & Summaries: reports/")
    print(" - Serialized Model: models/weather_classifier_pipeline.joblib")
    print("=" * 80)


if __name__ == "__main__":
    main()
