"""Model Training, Hyperparameter Tuning, and Comparison Pipeline.

Trains and compares multiple classifiers:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- HistGradientBoosting Classifier
- Support Vector Machine (SVC)

Performs 5-fold Stratified Cross-Validation, selects the champion model,
tunes hyperparameters, generates evaluation plots, and saves the final production pipeline.
"""

import os
import json
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report

from src.preprocess import prepare_data, engineer_features, clip_physical_outliers
from src.evaluate import (
    compute_metrics, plot_confusion_matrix, plot_multiclass_roc_curve,
    plot_model_comparison_bar, plot_feature_importance
)


def get_candidate_models() -> Dict[str, Any]:
    """Define candidate classification models with reproducible seeds."""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=14, random_state=42, n_jobs=-1),
        'Gradient Boosting': HistGradientBoostingClassifier(max_iter=150, max_depth=8, random_state=42),
        'Support Vector Machine': SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    }


def train_and_compare_models(data_bundle: Dict[str, Any], reports_dir: str = "reports") -> Tuple[pd.DataFrame, Any, str]:
    """Train candidate models, perform CV, and evaluate on held-out test set."""
    X_train = data_bundle['X_train_proc']
    y_train = data_bundle['y_train']
    X_test = data_bundle['X_test_proc']
    y_test = data_bundle['y_test']
    class_names = data_bundle['class_names']
    
    models = get_candidate_models()
    metrics_list = []
    trained_models = {}
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n" + "=" * 60)
    print("STAGE 1: MODEL TRAINING & 5-FOLD CROSS-VALIDATION")
    print("=" * 60)
    
    for name, model in models.items():
        print(f"\n[Training] {name}...")
        # 5-fold CV on train set
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Fit on full train set
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict on test set
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = np.zeros((len(y_test), len(class_names)))
            
        metrics = compute_metrics(y_test, y_pred, y_proba, name, class_names)
        metrics['CV Accuracy (Mean)'] = round(cv_scores.mean(), 4)
        metrics['CV Accuracy (Std)'] = round(cv_scores.std(), 4)
        metrics_list.append(metrics)
        print(f"  Test Accuracy: {metrics['Accuracy']:.4f} | F1 (Macro): {metrics['F1-Score (Macro)']:.4f}")
        
    comparison_df = pd.DataFrame(metrics_list).sort_values(by='F1-Score (Macro)', ascending=False).reset_index(drop=True)
    
    # Save comparison table
    os.makedirs(reports_dir, exist_ok=True)
    comparison_csv = os.path.join(reports_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"\nModel comparison metrics saved to: {comparison_csv}")
    
    # Select Champion Model
    champion_name = comparison_df.iloc[0]['Model']
    champion_model = trained_models[champion_name]
    print(f"\n>>> CHAMPION MODEL SELECTED: {champion_name} (F1-Macro = {comparison_df.iloc[0]['F1-Score (Macro)']}) <<<")
    
    return comparison_df, champion_model, champion_name


def tune_champion_model(champion_model: Any, champion_name: str, data_bundle: Dict[str, Any]) -> Any:
    """Perform Hyperparameter Optimization on the top-performing model."""
    print("\n" + "=" * 60)
    print(f"STAGE 2: HYPERPARAMETER TUNING FOR [{champion_name}]")
    print("=" * 60)
    
    X_train = data_bundle['X_train_proc']
    y_train = data_bundle['y_train']
    
    param_grid = {}
    if "Random Forest" in champion_name:
        param_grid = {
            'n_estimators': [100, 150, 200],
            'max_depth': [12, 16, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    elif "Gradient" in champion_name:
        param_grid = {
            'max_iter': [100, 150, 200],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [6, 8, 10],
            'l2_regularization': [0.0, 0.1]
        }
    else:
        print("Using default champion parameters (no custom grid defined).")
        return champion_model
        
    grid_search = GridSearchCV(
        estimator=champion_model,
        param_grid=param_grid,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


def save_champion_bundle(
    tuned_model: Any,
    data_bundle: Dict[str, Any],
    champion_name: str,
    output_path: str = "models/weather_classifier_pipeline.joblib"
) -> None:
    """Save full inference package containing preprocessor, model, and metadata."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    bundle = {
        'model': tuned_model,
        'model_name': champion_name,
        'preprocessor': data_bundle['preprocessor'],
        'label_encoder': data_bundle['label_encoder'],
        'class_names': data_bundle['class_names'],
        'feature_names': data_bundle['feature_names'],
        'num_cols': data_bundle['num_cols'],
        'cat_cols': data_bundle['cat_cols']
    }
    
    joblib.dump(bundle, output_path)
    print(f"\nProduction inference bundle successfully saved to: {output_path}")


def run_training_pipeline(data_path: str = "data/weather_classification_data.csv") -> None:
    """Orchestrate end-to-end model training, tuning, evaluation, and serialization."""
    reports_dir = "reports"
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Prepare data
    data_bundle = prepare_data(data_path)
    
    # 2. Train & Compare models
    comparison_df, champion_model, champion_name = train_and_compare_models(data_bundle, reports_dir)
    
    # 3. Plot model comparison
    plot_model_comparison_bar(comparison_df, os.path.join(figures_dir, "06_model_comparison.png"))
    
    # 4. Tune champion model
    tuned_model = tune_champion_model(champion_model, champion_name, data_bundle)
    
    # 5. Final Evaluation on Test Set
    X_test = data_bundle['X_test_proc']
    y_test = data_bundle['y_test']
    class_names = data_bundle['class_names']
    
    y_pred = tuned_model.predict(X_test)
    y_proba = tuned_model.predict_proba(X_test) if hasattr(tuned_model, 'predict_proba') else None
    
    print("\n" + "=" * 60)
    print(f"FINAL EVALUATION ON TEST SET ({champion_name} Tuned)")
    print("=" * 60)
    rep = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print(rep)
    
    # Save text report
    with open(os.path.join(reports_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"MODEL: {champion_name} (Tuned)\n")
        f.write("=" * 60 + "\n")
        f.write(rep + "\n\n")
        f.write("Model Comparison Matrix:\n")
        f.write(comparison_df.to_string())
        
    # 6. Evaluation Plots
    plot_confusion_matrix(
        y_test, y_pred, class_names, f"{champion_name} (Tuned)",
        os.path.join(figures_dir, "07_confusion_matrix_best_model.png")
    )
    
    if y_proba is not None:
        plot_multiclass_roc_curve(
            y_test, y_proba, class_names, f"{champion_name} (Tuned)",
            os.path.join(figures_dir, "08_multiclass_roc_best_model.png")
        )
        
    if hasattr(tuned_model, 'feature_importances_'):
        plot_feature_importance(
            data_bundle['feature_names'], tuned_model.feature_importances_,
            champion_name, os.path.join(figures_dir, "09_feature_importances.png")
        )
        
    # 7. Save production bundle
    save_champion_bundle(tuned_model, data_bundle, champion_name)
    
    print("\n" + "=" * 60)
    print("PIPELINE TRAINING & EVALUATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    run_training_pipeline()
