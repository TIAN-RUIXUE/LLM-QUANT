# scripts/generate_shap_figure.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yfinance as yf
import matplotlib.pyplot as plt
import shap
import numpy as np
from sklearn.preprocessing import StandardScaler

from strategy.quant_model_old import (
    compute_technical_features, prepare_ml_features, train_xgb_model
)

# fetch data
raw = yf.download('D05.SI', start='2024-09-24', end='2026-09-24', progress=False)
featured = compute_technical_features(raw)

# train model
model, X_train, X_test, y_train, y_test, scaler = train_xgb_model(featured)

# prepare features
feature_names = prepare_ml_features(featured).columns.tolist()

# calculate SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# draw maps
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values, 
    X_test, 
    feature_names=feature_names,
    show=False,
    plot_type="bar"
)
plt.tight_layout()

# save
output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                          "..", "LatexTemplate", "figures")
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "shap_importance.pdf")
plt.savefig(save_path, bbox_inches='tight')
print(f"Figure saved to {save_path}")
plt.close()