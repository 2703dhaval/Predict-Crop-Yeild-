"""
🌾 Crop Yield Prediction using Machine Learning
---------------------------------------------
This script implements a complete end-to-end Machine Learning pipeline to predict 
crop yield using Random Forest Regression and Gradient Boosting Regression.

It contains 13 detailed steps:
1. Import Libraries
2. Load Dataset
3. Exploratory Data Analysis (EDA)
4. Data Visualization (Exploratory Plots)
5. Data Preprocessing (Label Encoding & Target Splitting)
6. Train-Test Split (80/20)
7. Train & Evaluate Random Forest Regressor
8. Train & Evaluate Gradient Boosting Regressor
9. Model Comparison & Metrics Plotting
10. Actual vs Predicted Analysis & Plotting
11. Feature Importance Analysis
12. Demo: Predict on New Data
13. Residual (Error) Analysis
"""

# =====================================================================
# Step 1: Import Libraries
# =====================================================================
print("\n--- Step 1: Importing Necessary Libraries ---")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML Preprocessing, Modeling, and Evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set visualization aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# =====================================================================
# Step 2: Load Dataset
# =====================================================================
print("\n--- Step 2: Loading the Kaggle Dataset ---")
csv_path = "crop_yield_data.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset file '{csv_path}' not found in the workspace!")

df = pd.read_csv(csv_path)
print(f"Dataset successfully loaded. Shape: {df.shape[0]} rows, {df.shape[1]} columns")

print("\nDataset Info Summary:")
df.info()

print("\nFirst 5 rows of the dataset:")
print(df.head())

# =====================================================================
# Step 3: Exploratory Data Analysis (EDA)
# =====================================================================
print("\n--- Step 3: Exploratory Data Analysis (EDA) ---")

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values per column:")
print(missing_values)

# Drop any row containing missing values or handle them robustly
if missing_values.sum() > 0:
    print("Handling missing values by dropping affected rows...")
    df.dropna(inplace=True)
    print(f"New shape after cleaning: {df.shape}")
else:
    print("Perfect! No missing values found in the dataset.")

# Stripping whitespace from categorical columns to avoid encoding issues
for col in ['Crop', 'Season', 'State']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

print("\nSummary Statistics of Numeric Columns:")
print(df.describe())

# =====================================================================
# Step 4: Data Visualization (Exploratory Plots)
# =====================================================================
print("\n--- Step 4: Generating Exploratory Visualizations ---")

# 1. Distribution of Target Variable 'Yield'
plt.figure(figsize=(10, 6))
sns.histplot(df['Yield'], kde=True, bins=50, color='#1f77b4')
plt.title('Distribution of Crop Yield (Tons per Hectare)', pad=15)
plt.xlabel('Yield')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('plot_1_yield_distribution.png', dpi=150)
plt.close()
print("Saved: plot_1_yield_distribution.png")

# 2. Correlation Heatmap of Numeric Features
plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar=True)
plt.title('Correlation Heatmap of Numeric Features', pad=15)
plt.tight_layout()
plt.savefig('plot_2_correlation_heatmap.png', dpi=150)
plt.close()
print("Saved: plot_2_correlation_heatmap.png")

# 3. Bar Plot: Average Yield of Top 10 Crops
plt.figure(figsize=(12, 6))
top_crops = df.groupby('Crop')['Yield'].mean().sort_values(ascending=False).head(10).reset_index()
sns.barplot(x='Yield', y='Crop', data=top_crops, palette='viridis', hue='Crop', legend=False)
plt.title('Top 10 Crops by Average Yield (Tons per Hectare)', pad=15)
plt.xlabel('Average Yield')
plt.ylabel('Crop Name')
plt.tight_layout()
plt.savefig('plot_3_top_crops.png', dpi=150)
plt.close()
print("Saved: plot_3_top_crops.png")

# 4. Scatter Plot: Annual Rainfall vs Crop Yield
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual_Rainfall', y='Yield', data=df, alpha=0.5, color='#e377c2')
plt.title('Annual Rainfall vs Crop Yield', pad=15)
plt.xlabel('Annual Rainfall (mm)')
plt.ylabel('Yield')
plt.tight_layout()
plt.savefig('plot_4_rainfall_vs_yield.png', dpi=150)
plt.close()
print("Saved: plot_4_rainfall_vs_yield.png")

# 5. Scatter Plot: Fertilizer Used vs Crop Yield
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Fertilizer', y='Yield', data=df, alpha=0.5, color='#2ca02c')
plt.title('Fertilizer Usage vs Crop Yield', pad=15)
plt.xlabel('Fertilizer Used (kg)')
plt.ylabel('Yield')
plt.tight_layout()
plt.savefig('plot_5_fertilizer_vs_yield.png', dpi=150)
plt.close()
print("Saved: plot_5_fertilizer_vs_yield.png")

# 6. Line Plot: Average Crop Yield Trend Over Years
plt.figure(figsize=(10, 6))
yield_trend = df.groupby('Crop_Year')['Yield'].mean().reset_index()
sns.lineplot(x='Crop_Year', y='Yield', data=yield_trend, marker='o', color='#ff7f0e', linewidth=2.5)
plt.title('Average Crop Yield Trend (1997 - 2020)', pad=15)
plt.xlabel('Crop Year')
plt.ylabel('Average Yield')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plot_6_yield_over_years.png', dpi=150)
plt.close()
print("Saved: plot_6_yield_over_years.png")

# =====================================================================
# Step 5: Data Preprocessing
# =====================================================================
print("\n--- Step 5: Data Preprocessing & Categorical Encoding ---")

# Let's perform a drop of the 'Production' column to avoid target leakage,
# as Yield = Production / Area.
print("Dropping 'Production' to prevent data leakage...")
df_processed = df.copy()

# Keep track of LabelEncoders for new data predictions
label_encoders = {}
cat_cols = ['Crop', 'Season', 'State']

for col in cat_cols:
    print(f"Label Encoding column: {col}")
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le

# Separate features (X) and target (y)
X = df_processed.drop(['Yield', 'Production'], axis=1)
y = df_processed['Yield']

print(f"\nFeatures selected: {list(X.columns)}")
print(f"Target selected: Yield")

# =====================================================================
# Step 6: Train-Test Split
# =====================================================================
print("\n--- Step 6: Train-Test Split (80/20) ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} rows")
print(f"Testing set size: {X_test.shape[0]} rows")

# =====================================================================
# Step 7: Train Random Forest Regressor
# =====================================================================
print("\n--- Step 7: Training Random Forest Regressor ---")
# 100 decision trees in the forest, keeping random_state fixed for reproducibility
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Evaluate Random Forest
y_pred_rf = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_r2 = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Results on Test Set:")
print(f"Mean Absolute Error (MAE): {rf_mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rf_rmse:.4f}")
print(f"R^2 Score (Accuracy): {rf_r2:.4f}")

# =====================================================================
# Step 8: Train Gradient Boosting Regressor
# =====================================================================
print("\n--- Step 8: Training Gradient Boosting Regressor ---")
# Instantiate sequentially building trees with max depth 5 to capture complexities
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)

# Evaluate Gradient Boosting
y_pred_gb = gb_model.predict(X_test)
gb_mae = mean_absolute_error(y_test, y_pred_gb)
gb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_gb))
gb_r2 = r2_score(y_test, y_pred_gb)

print("\nGradient Boosting Results on Test Set:")
print(f"Mean Absolute Error (MAE): {gb_mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {gb_rmse:.4f}")
print(f"R^2 Score (Accuracy): {gb_r2:.4f}")

# =====================================================================
# Step 9: Model Comparison & Metrics Plotting
# =====================================================================
print("\n--- Step 9: Model Performance Comparison ---")
comparison_df = pd.DataFrame({
    'Metric': ['MAE (Lower is Better)', 'RMSE (Lower is Better)', 'R^2 Score (Higher is Better)'],
    'Random Forest': [rf_mae, rf_rmse, rf_r2],
    'Gradient Boosting': [gb_mae, gb_rmse, gb_r2]
})
print("\n" + "=" * 55)
print(comparison_df.to_string(index=False))
print("=" * 55)

# Determine the winner
best_model_name = "Gradient Boosting" if gb_r2 > rf_r2 else "Random Forest"
print(f"\nWINNER: {best_model_name} Regressor (higher R^2 Score of {max(rf_r2, gb_r2):.4f})")

# Save plot_7_model_comparison.png
plt.figure(figsize=(10, 6))
metrics_plot_data = pd.melt(comparison_df, id_vars=['Metric'], value_vars=['Random Forest', 'Gradient Boosting'],
                            var_name='Model', value_name='Value')

# Filter R2 score and Error metrics separately for visualization scale
sns.barplot(x='Metric', y='Value', hue='Model', data=metrics_plot_data, palette='Set2')
plt.title('Performance Comparison: Random Forest vs Gradient Boosting', pad=15)
plt.ylabel('Metric Score')
plt.xlabel('')
plt.legend(title='Models')
plt.tight_layout()
plt.savefig('plot_7_model_comparison.png', dpi=150)
plt.close()
print("Saved: plot_7_model_comparison.png")

# =====================================================================
# Step 10: Actual vs Predicted Analysis
# =====================================================================
print("\n--- Step 10: Generating Actual vs Predicted Plot ---")
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# Random Forest Actual vs Predicted
sns.scatterplot(x=y_test, y=y_pred_rf, ax=axes[0], alpha=0.6, color='teal')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2.5)
axes[0].set_title(f'Random Forest (R^2 = {rf_r2:.4f})')
axes[0].set_xlabel('Actual Yield')
axes[0].set_ylabel('Predicted Yield')

# Gradient Boosting Actual vs Predicted
sns.scatterplot(x=y_test, y=y_pred_gb, ax=axes[1], alpha=0.6, color='coral')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2.5)
axes[1].set_title(f'Gradient Boosting (R^2 = {gb_r2:.4f})')
axes[1].set_xlabel('Actual Yield')

plt.suptitle('Actual vs Predicted Yield (Tons per Hectare)', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('plot_8_actual_vs_predicted.png', dpi=150)
plt.close()
print("Saved: plot_8_actual_vs_predicted.png")

# =====================================================================
# Step 11: Feature Importance Analysis
# =====================================================================
print("\n--- Step 11: Feature Importance Analysis ---")
features = X.columns
rf_importances = rf_model.feature_importances_
gb_importances = gb_model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': features,
    'Random Forest': rf_importances,
    'Gradient Boosting': gb_importances
}).sort_values(by='Random Forest', ascending=False)

print("\nFeature Importance Scores:")
print(importance_df.to_string(index=False))

# Plot Feature Importances
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# RF Importances
sns.barplot(x='Random Forest', y='Feature', data=importance_df.sort_values(by='Random Forest', ascending=False),
            ax=axes[0], palette='Blues_r', hue='Feature', legend=False)
axes[0].set_title('Random Forest Feature Importance')
axes[0].set_xlabel('Importance Score')

# GB Importances
sns.barplot(x='Gradient Boosting', y='Feature', data=importance_df.sort_values(by='Gradient Boosting', ascending=False),
            ax=axes[1], palette='Oranges_r', hue='Feature', legend=False)
axes[1].set_title('Gradient Boosting Feature Importance')
axes[1].set_xlabel('Importance Score')
axes[1].set_ylabel('')

plt.suptitle('Model Feature Importance Comparison', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('plot_9_feature_importance.png', dpi=150)
plt.close()
print("Saved: plot_9_feature_importance.png")

# =====================================================================
# Step 12: Demo: Predict on New Data
# =====================================================================
print("\n--- Step 12: Demo - Predicting Crop Yield on New Input Data ---")

# Let's select an actual row from the dataset for testing or use custom inputs
sample_idx = 0
sample_row = df.iloc[sample_idx]

print("\n--- Demo Input Data (Original Text format) ---")
for col in df.columns:
    if col not in ['Yield', 'Production']:
        print(f"{col}: {sample_row[col]}")

# Create a dictionary to hold our new input data
new_data_dict = {
    'Crop': sample_row['Crop'],
    'Crop_Year': sample_row['Crop_Year'],
    'Season': sample_row['Season'],
    'State': sample_row['State'],
    'Area': sample_row['Area'],
    'Annual_Rainfall': sample_row['Annual_Rainfall'],
    'Fertilizer': sample_row['Fertilizer'],
    'Pesticide': sample_row['Pesticide']
}

# Construct dataframe for prediction
new_df = pd.DataFrame([new_data_dict])

# Encode categorical variables using trained LabelEncoders
try:
    for col in cat_cols:
        new_df[col] = label_encoders[col].transform(new_df[col])
    
    # Predict using both models
    rf_pred = rf_model.predict(new_df)[0]
    gb_pred = gb_model.predict(new_df)[0]
    actual_val = sample_row['Yield']
    
    print("\n--- Predictions ---")
    print(f"Random Forest Predicted Yield: {rf_pred:.4f} tons/hectare")
    print(f"Gradient Boosting Predicted Yield: {gb_pred:.4f} tons/hectare")
    print(f"Actual Yield: {actual_val:.4f} tons/hectare")
    print(f"Random Forest Error: {abs(rf_pred - actual_val):.4f}")
    print(f"Gradient Boosting Error: {abs(gb_pred - actual_val):.4f}")
except Exception as e:
    print(f"Error encoding/predicting new data: {e}")

# =====================================================================
# Step 13: Residual (Error) Analysis
# =====================================================================
print("\n--- Step 13: Residual (Error) Analysis ---")
rf_residuals = y_test - y_pred_rf
gb_residuals = y_test - y_pred_gb

plt.figure(figsize=(12, 6))
sns.kdeplot(rf_residuals, fill=True, color='teal', label=f'Random Forest (Mean Error = {rf_residuals.mean():.4f})', alpha=0.5)
sns.kdeplot(gb_residuals, fill=True, color='coral', label=f'Gradient Boosting (Mean Error = {gb_residuals.mean():.4f})', alpha=0.5)
plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error')
plt.title('Residuals (Errors) Distribution Comparison', pad=15)
plt.xlabel('Residual Value (Actual - Predicted)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('plot_10_residual_analysis.png', dpi=150)
plt.close()
print("Saved: plot_10_residual_analysis.png")

print("\n*** ML Crop Yield Prediction script completed successfully! ***")
print("All 10 visualization plots are saved in the project folder.\n")
