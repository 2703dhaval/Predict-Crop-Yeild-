"""
🌾 Crop Yield Prediction - Interactive Flask Web Application
-----------------------------------------------------------
This file serves as the web server controller (backend) for our Machine Learning project.
It integrates dataset loading, ensemble regressor training, visualization plotting,
and real-time user-input forecasting into a clean, single-purpose dashboard.
"""

import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
# Force matplotlib to run in a non-GUI backend since we are running on a server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize the Flask Web App
app = Flask(__name__)

# Base directory setup for absolute path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_CSV_PATH = os.path.join(BASE_DIR, "crop_yield_data.csv")

# Global variables to store trained models, encoders, and performance scores
MODELS = {
    'rf': None,
    'gb': None
}
LABEL_ENCODERS = {}
FEATURES_LIST = []
METRICS = {
    'rf': {'mae': 0, 'rmse': 0, 'r2': 0},
    'gb': {'mae': 0, 'rmse': 0, 'r2': 0}
}
PLOTS_BASE64 = {}
DROPDOWN_OPTIONS = {
    'crops': [],
    'seasons': [],
    'states': []
}
WINNER = ""
DATASET_PREVIEW = ""
DATASET_SHAPE = (0, 0)

# Set Seaborn styling parameters
sns.set_theme(style="whitegrid")

def fig_to_base64(fig):
    """Utility function to convert matplotlib figures into base64 strings for inline HTML rendering."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_str

def train_and_evaluate_models(csv_path):
    """
    Executes the complete machine learning pipeline:
    1. Loads dataset
    2. Handles LabelEncoding
    3. Triggers train-test split (80/20)
    4. Trains RF and GB models
    5. Generates metrics and comparison plots
    """
    global MODELS, LABEL_ENCODERS, FEATURES_LIST, METRICS, PLOTS_BASE64, DROPDOWN_OPTIONS, WINNER, DATASET_PREVIEW, DATASET_SHAPE
    
    # 1. Load Dataset
    df = pd.read_csv(csv_path)
    
    # 2. Clean dataset (remove trailing whitespace in categorical strings)
    for col in ['Crop', 'Season', 'State']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
    # Remove null rows if any
    df.dropna(inplace=True)
    DATASET_SHAPE = df.shape
    
    # Create HTML table preview of first 5 rows
    DATASET_PREVIEW = df.head(10).to_html(classes='preview-table', index=False, border=0)
    
    # Store options for dynamic frontend dropdowns
    DROPDOWN_OPTIONS['crops'] = sorted(df['Crop'].unique().tolist())
    DROPDOWN_OPTIONS['seasons'] = sorted(df['Season'].unique().tolist())
    DROPDOWN_OPTIONS['states'] = sorted(df['State'].unique().tolist())
    
    # 3. Preprocessing (Label Encoding & Target Dropping)
    df_processed = df.copy()
    LABEL_ENCODERS = {}
    
    for col in ['Crop', 'Season', 'State']:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        LABEL_ENCODERS[col] = le
        
    X = df_processed.drop(['Yield', 'Production'], axis=1)
    y = df_processed['Yield']
    FEATURES_LIST = list(X.columns)
    
    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    # 6. Train Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    
    # Store models globally
    MODELS['rf'] = rf
    MODELS['gb'] = gb
    
    # 7. Evaluate Metrics
    METRICS['rf']['mae'] = mean_absolute_error(y_test, y_pred_rf)
    METRICS['rf']['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    METRICS['rf']['r2'] = r2_score(y_test, y_pred_rf)
    
    METRICS['gb']['mae'] = mean_absolute_error(y_test, y_pred_gb)
    METRICS['gb']['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred_gb))
    METRICS['gb']['r2'] = r2_score(y_test, y_pred_gb)
    
    WINNER = "Random Forest Regressor" if METRICS['rf']['r2'] > METRICS['gb']['r2'] else "Gradient Boosting Regressor"
    
    # 8. Generate Visualizations (Base64 conversion)
    PLOTS_BASE64 = {}
    
    # Plot 1: Distribution of Crop Yield
    fig = plt.figure(figsize=(9, 5))
    sns.histplot(df['Yield'], kde=True, bins=50, color='#14b8a6')
    plt.title('Distribution of Crop Yield (Tons per Hectare)', pad=12, fontsize=12, fontweight='bold', color='#ffffff')
    plt.xlabel('Yield', color='#ffffff')
    plt.ylabel('Frequency', color='#ffffff')
    fig.patch.set_facecolor('#0f172a')
    plt.gca().set_facecolor('#1e293b')
    plt.tick_params(colors='#ffffff')
    PLOTS_BASE64['plot_1'] = fig_to_base64(fig)
    
    # Plot 2: Correlation Heatmap
    fig = plt.figure(figsize=(9, 6))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar=True)
    plt.title('Correlation Heatmap of Numeric Parameters', pad=12, fontsize=12, fontweight='bold', color='#ffffff')
    fig.patch.set_facecolor('#0f172a')
    plt.tick_params(colors='#ffffff')
    PLOTS_BASE64['plot_2'] = fig_to_base64(fig)
    
    # Plot 3: Top Crops by Average Yield
    fig = plt.figure(figsize=(9, 5))
    top_crops = df.groupby('Crop')['Yield'].mean().sort_values(ascending=False).head(10).reset_index()
    sns.barplot(x='Yield', y='Crop', data=top_crops, palette='crest', hue='Crop', legend=False)
    plt.title('Top 10 High-Yielding Crops (Average)', pad=12, fontsize=12, fontweight='bold', color='#ffffff')
    plt.xlabel('Average Yield (Tons/Hectare)', color='#ffffff')
    plt.ylabel('Crop Name', color='#ffffff')
    fig.patch.set_facecolor('#0f172a')
    plt.gca().set_facecolor('#1e293b')
    plt.tick_params(colors='#ffffff')
    PLOTS_BASE64['plot_3'] = fig_to_base64(fig)
    
    # Plot 4: Annual Rainfall vs Crop Yield
    fig = plt.figure(figsize=(9, 5))
    sns.scatterplot(x='Annual_Rainfall', y='Yield', data=df, alpha=0.5, color='#a855f7')
    plt.title('Annual Rainfall (mm) vs Crop Yield', pad=12, fontsize=12, fontweight='bold', color='#ffffff')
    plt.xlabel('Annual Rainfall (mm)', color='#ffffff')
    plt.ylabel('Yield', color='#ffffff')
    fig.patch.set_facecolor('#0f172a')
    plt.gca().set_facecolor('#1e293b')
    plt.tick_params(colors='#ffffff')
    PLOTS_BASE64['plot_4'] = fig_to_base64(fig)
    
    # Plot 5: Fertilizer Used vs Crop Yield
    fig = plt.figure(figsize=(9, 5))
    sns.scatterplot(x='Fertilizer', y='Yield', data=df, alpha=0.5, color='#22c55e')
    plt.title('Fertilizer Usage (kg) vs Crop Yield', pad=12, fontsize=12, fontweight='bold', color='#ffffff')
    plt.xlabel('Fertilizer Used (kg)', color='#ffffff')
    plt.ylabel('Yield', color='#ffffff')
    fig.patch.set_facecolor('#0f172a')
    plt.gca().set_facecolor('#1e293b')
    plt.tick_params(colors='#ffffff')
    PLOTS_BASE64['plot_5'] = fig_to_base64(fig)
    
    # Plot 6: Average Crop Yield Trend Over Years
    fig = plt.figure(figsize=(9, 5))
    yield_trend = df.groupby('Crop_Year')['Yield'].mean().reset_index()
    sns.lineplot(x='Crop_Year', y='Yield', data=yield_trend, marker='o', color='#e11d48', linewidth=2.5)
    plt.title('Average Yield Trend (1997 - 2020)', pad=12, fontsize=12, fontweight='bold', color='#ffffff')
    plt.xlabel('Crop Year', color='#ffffff')
    plt.ylabel('Average Yield', color='#ffffff')
    fig.patch.set_facecolor('#0f172a')
    plt.gca().set_facecolor('#1e293b')
    plt.tick_params(colors='#ffffff')
    PLOTS_BASE64['plot_6'] = fig_to_base64(fig)
    
    # Plot 7: Model Performance Comparison
    fig = plt.figure(figsize=(9, 5))
    comp_df = pd.DataFrame({
        'Metric': ['MAE', 'MAE', 'RMSE', 'RMSE', 'R2 Score', 'R2 Score'],
        'Model': ['Random Forest', 'Gradient Boosting', 'Random Forest', 'Gradient Boosting', 'Random Forest', 'Gradient Boosting'],
        'Value': [METRICS['rf']['mae'], METRICS['gb']['mae'], METRICS['rf']['rmse'], METRICS['gb']['rmse'], METRICS['rf']['r2'], METRICS['gb']['r2']]
    })
    # Filter only R2 score and MAE for visualization scale clarity
    sns.barplot(x='Metric', y='Value', hue='Model', data=comp_df[comp_df['Metric'] != 'RMSE'], palette='Set2')
    plt.title('Algorithm Performance Comparison (MAE & R2)', pad=12, fontsize=12, fontweight='bold', color='#ffffff')
    plt.ylabel('Score / Error', color='#ffffff')
    plt.xlabel('')
    fig.patch.set_facecolor('#0f172a')
    plt.gca().set_facecolor('#1e293b')
    plt.tick_params(colors='#ffffff')
    plt.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#ffffff')
    PLOTS_BASE64['plot_7'] = fig_to_base64(fig)
    
    # Plot 8: Actual vs Predicted Yield
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    sns.scatterplot(x=y_test, y=y_pred_rf, ax=axes[0], alpha=0.5, color='#06b6d4')
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2.0)
    axes[0].set_title('Random Forest Forecast', color='#ffffff', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Actual Yield', color='#ffffff')
    axes[0].set_ylabel('Predicted Yield', color='#ffffff')
    axes[0].tick_params(colors='#ffffff')
    axes[0].set_facecolor('#1e293b')
    
    sns.scatterplot(x=y_test, y=y_pred_gb, ax=axes[1], alpha=0.5, color='#f97316')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2.0)
    axes[1].set_title('Gradient Boosting Forecast', color='#ffffff', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Actual Yield', color='#ffffff')
    axes[1].tick_params(colors='#ffffff')
    axes[1].set_facecolor('#1e293b')
    
    fig.patch.set_facecolor('#0f172a')
    plt.suptitle('Actual vs Predicted Yield Comparison', fontsize=13, fontweight='bold', color='#ffffff', y=0.98)
    plt.tight_layout()
    PLOTS_BASE64['plot_8'] = fig_to_base64(fig)
    
    # Plot 9: Feature Importance
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    imp_df = pd.DataFrame({
        'Feature': FEATURES_LIST,
        'Random Forest': rf.feature_importances_,
        'Gradient Boosting': gb.feature_importances_
    }).sort_values(by='Random Forest', ascending=False)
    
    sns.barplot(x='Random Forest', y='Feature', data=imp_df, ax=axes[0], palette='Blues_r', hue='Feature', legend=False)
    axes[0].set_title('Random Forest Feature Weights', color='#ffffff', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Weight Score', color='#ffffff')
    axes[0].set_ylabel('Features', color='#ffffff')
    axes[0].tick_params(colors='#ffffff')
    axes[0].set_facecolor('#1e293b')
    
    sns.barplot(x='Gradient Boosting', y='Feature', data=imp_df.sort_values(by='Gradient Boosting', ascending=False), ax=axes[1], palette='Oranges_r', hue='Feature', legend=False)
    axes[1].set_title('Gradient Boosting Feature Weights', color='#ffffff', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Weight Score', color='#ffffff')
    axes[1].set_ylabel('')
    axes[1].tick_params(colors='#ffffff')
    axes[1].set_facecolor('#1e293b')
    
    fig.patch.set_facecolor('#0f172a')
    plt.suptitle('Predictive Feature Importance Comparison', fontsize=13, fontweight='bold', color='#ffffff', y=0.98)
    plt.tight_layout()
    PLOTS_BASE64['plot_9'] = fig_to_base64(fig)
    
    # Plot 10: Residual analysis
    fig = plt.figure(figsize=(9, 5))
    rf_res = y_test - y_pred_rf
    gb_res = y_test - y_pred_gb
    sns.kdeplot(rf_res, fill=True, color='#06b6d4', label=f'Random Forest (Mean = {rf_res.mean():.3f})', alpha=0.4)
    sns.kdeplot(gb_res, fill=True, color='#f97316', label=f'Gradient Boosting (Mean = {gb_res.mean():.3f})', alpha=0.4)
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Optimal Zero-Error')
    plt.title('Residuals Error Distribution Comparison', pad=12, fontsize=12, fontweight='bold', color='#ffffff')
    plt.xlabel('Residual Value (Actual - Predicted)', color='#ffffff')
    plt.ylabel('Density', color='#ffffff')
    fig.patch.set_facecolor('#0f172a')
    plt.gca().set_facecolor('#1e293b')
    plt.tick_params(colors='#ffffff')
    plt.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#ffffff')
    PLOTS_BASE64['plot_10'] = fig_to_base64(fig)
    
    print("Models trained and dynamic base64 plots created successfully!")

# Train on startup using the absolute path
if os.path.exists(CURRENT_CSV_PATH):
    print("Initiating automatic model training on startup...")
    try:
        train_and_evaluate_models(CURRENT_CSV_PATH)
    except Exception as e:
        print(f"Startup training error: {e}")
else:
    print(f"Warning: CSV file not found at {CURRENT_CSV_PATH}.")

# =====================================================================
# HTTP Server Routes
# =====================================================================

@app.route('/')
def index():
    """Renders the main dashboard template containing state displays."""
    return render_template(
        'index.html',
        shape=DATASET_SHAPE,
        preview=DATASET_PREVIEW,
        metrics=METRICS,
        winner=WINNER,
        plots=PLOTS_BASE64,
        dropdowns=DROPDOWN_OPTIONS,
        csv_name=os.path.basename(CURRENT_CSV_PATH)
    )

@app.route('/train', methods=['POST'])
def trigger_training():
    """Forces retraining using the active CSV path."""
    try:
        train_and_evaluate_models(CURRENT_CSV_PATH)
        return jsonify({
            'success': True,
            'message': 'Models and visual plots re-evaluated successfully!'
        })
    except Exception as e:
        return jsonify({'error': f'Retraining failed: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict_yield():
    """Endpoint receiving manual data forms to evaluate yield projections."""
    if MODELS['rf'] is None or MODELS['gb'] is None:
        return jsonify({'error': 'Models are not trained. Please make sure the dataset is loaded.'}), 400
        
    try:
        # Collect request form inputs
        crop_val = request.form.get('crop')
        year_val = int(request.form.get('year'))
        season_val = request.form.get('season')
        state_val = request.form.get('state')
        area_val = float(request.form.get('area'))
        rainfall_val = float(request.form.get('rainfall'))
        fertilizer_val = float(request.form.get('fertilizer'))
        pesticide_val = float(request.form.get('pesticide'))
        
        # Construct prediction dataframe
        input_data = {
            'Crop': crop_val,
            'Crop_Year': year_val,
            'Season': season_val,
            'State': state_val,
            'Area': area_val,
            'Annual_Rainfall': rainfall_val,
            'Fertilizer': fertilizer_val,
            'Pesticide': pesticide_val
        }
        pred_df = pd.DataFrame([input_data])
        
        # Label encode the categorical inputs using the saved encoders
        for col in ['Crop', 'Season', 'State']:
            pred_df[col] = LABEL_ENCODERS[col].transform(pred_df[col])
            
        # Predict using both models
        rf_prediction = MODELS['rf'].predict(pred_df)[0]
        gb_prediction = MODELS['gb'].predict(pred_df)[0]
        
        return jsonify({
            'success': True,
            'rf_pred': round(rf_prediction, 4),
            'gb_pred': round(gb_prediction, 4),
            'crop': crop_val,
            'state': state_val
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed. Please verify input values: {str(e)}'}), 500

if __name__ == '__main__':
    # Start web app on localhost:5000
    app.run(debug=True, host='127.0.0.1', port=5000)
