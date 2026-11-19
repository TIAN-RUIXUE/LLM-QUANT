# strategy/shap_explainer.py

import shap
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def compute_shap_values(
    model: XGBClassifier,
    X_scaled: np.ndarray,
    feature_names: list
) -> np.ndarray:
    """
    Compute SHAP values.
    
    Parameters:
        model: trained XGBoost model
        X_scaled: standardized feature matrix
        feature_names: list of feature names
        
    Returns:
        shap_values: SHAP value matrix, shape (n_samples, n_features)
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    return shap_values


def get_base_value(model: XGBClassifier) -> float:
    """
    Return SHAP base value (in log-odds space).
    XGBoost binary classification TreeExplainer outputs log-odds contributions:
        prediction probability = sigmoid(base_value + sum(shap_values))
    """
    expected = shap.TreeExplainer(model).expected_value
    return float(np.ravel(expected)[0])


def explain_prediction(
    shap_values: np.ndarray,
    feature_names: list,
    X_row: pd.Series,
    top_k: int = 5,
    base_value: float = 0.0
) -> dict:
    """
    Explain SHAP attributions for a single prediction.
    
    Parameters:
        shap_values: SHAP values for this sample (n_features,)
        feature_names: list of feature names
        X_row: feature values for this sample (pd.Series)
        top_k: return top k most important features
        
    Returns:
        {
            "top_features": [
                {"name": "rsi", "value": 45.3, "shap": 0.12, "impact": "positive"},
                ...
            ],
            "base_value": 0.5,  # base value (mean prediction probability across all samples)
            "prediction": 0.75   # predicted probability for this sample
        }
    """
    # Get top-k features
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[-top_k:][::-1]
    
    top_features = []
    for idx in top_indices:
        impact = "positive" if shap_values[idx] > 0 else "negative"
        top_features.append({
            "name": feature_names[idx],
            "value": float(X_row.iloc[idx]),
            "shap": float(shap_values[idx]),
            "impact": impact
        })
    
    # SHAP values are in log-odds space: probability = sigmoid(base_value + sum(shap_values))
    logit = base_value + float(np.sum(shap_values))
    return {
        "top_features": top_features,
        "base_value": float(1.0 / (1.0 + np.exp(-base_value))),
        "prediction": float(1.0 / (1.0 + np.exp(-logit)))
    }


def plot_shap_summary(shap_values: np.ndarray, feature_names: list, save_path: str = None):
    """
    Plot global SHAP importance summary.
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def get_feature_importance(shap_values: np.ndarray, feature_names: list) -> pd.DataFrame:
    """
    Get global feature importance (sorted by mean absolute SHAP value).
    """
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_abs_shap
    }).sort_values("importance", ascending=False)
    return importance_df


if __name__ == "__main__":
    import yfinance as yf
    from strategy.quant_model_old import (
        compute_technical_features, prepare_ml_features,
        train_xgb_model, generate_ml_signals
    )
    
    # Fetch data
    raw = yf.download('D05.SI', start='2024-09-24', end='2026-09-24', progress=False)
    featured = compute_technical_features(raw)
    
    # Train model
    model, X_train, X_test, y_train, y_test, scaler = train_xgb_model(featured)
    
    # Compute SHAP
    feature_names = prepare_ml_features(featured).columns.tolist()
    shap_values = compute_shap_values(model, X_test, feature_names)
    
    # Global feature importance
    importance_df = get_feature_importance(shap_values, feature_names)
    print("\nGlobal Feature Importance (sorted by mean |SHAP|):")
    print(importance_df.head(10))
    
    # Explain the last test sample
    print("\nSHAP Explanation for the Last Test Sample:")
    sample_idx = -1
    explanation = explain_prediction(
        shap_values[sample_idx],
        feature_names,
        pd.Series(X_test[sample_idx], index=feature_names),
        base_value=get_base_value(model)
    )
    print(f"Predicted Probability: {explanation['prediction']:.4f}")
    print("Top Features:")
    for feat in explanation['top_features']:
        arrow = "↑" if feat['impact'] == 'positive' else "↓"
        print(f"  {feat['name']}: {feat['value']:.4f} -> SHAP={feat['shap']:.4f} {arrow}")