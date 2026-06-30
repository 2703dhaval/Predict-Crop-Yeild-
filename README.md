# 🌾 Crop Yield Prediction using Machine Learning

## 📌 Project Overview
This project predicts **crop yield** using two Machine Learning regression models:

1. **Random Forest Regression**
2. **Gradient Boosting Regression**

We use a **real-world Kaggle dataset** of agricultural crop yield across Indian states (1997–2020) and compare both models to find which gives better predictions.

---

## 📂 Dataset

- **Source**: [Kaggle – Agricultural Crop Yield in Indian States Dataset](https://www.kaggle.com/datasets/akshatgupta7/crop-yield-in-indian-states-dataset)
- **File**: `crop_yield_data.csv`
- **Records**: ~20,000 rows
- **Time Period**: 1997 – 2020

### Dataset Columns

| Column           | Description                              | Type        |
|------------------|------------------------------------------|-------------|
| Crop             | Name of the crop (Rice, Wheat, etc.)     | Categorical |
| Crop_Year        | Year when crop was grown                 | Numerical   |
| Season           | Kharif / Rabi / Whole Year / Summer etc. | Categorical |
| State            | Indian state (Punjab, Maharashtra, etc.) | Categorical |
| Area             | Land area under cultivation (hectares)   | Numerical   |
| Production       | Quantity of crop produced (metric tons)  | Numerical   |
| Annual_Rainfall  | Rainfall received (mm)                   | Numerical   |
| Fertilizer       | Fertilizer used (kg)                     | Numerical   |
| Pesticide        | Pesticide used (kg)                      | Numerical   |
| **Yield**        | **Target – crop yield (production/area)**| Numerical   |

---

## 📂 Project Structure

```
crop-yield-prediction/
│
├── crop_yield_data.csv           # Real Kaggle dataset
├── crop_yield_prediction.py      # Main ML script (13 steps)
├── generate_dataset.py           # (Optional) Generate sample data
├── requirements.txt              # Required Python packages
├── README.md                     # This file
├── VIVA_QUESTIONS.md             # Viva Q&A for exam prep
│
└── (plots generated after running)
    ├── plot_1_yield_distribution.png
    ├── plot_2_correlation_heatmap.png
    ├── plot_3_top_crops.png
    ├── plot_4_rainfall_vs_yield.png
    ├── plot_5_fertilizer_vs_yield.png
    ├── plot_6_yield_over_years.png
    ├── plot_7_model_comparison.png
    ├── plot_8_actual_vs_predicted.png
    ├── plot_9_feature_importance.png
    └── plot_10_residual_analysis.png
```

---

## 🛠️ How to Run

### Step 1: Install required packages
```bash
pip install -r requirements.txt
```

### Step 2: Run the prediction script
```bash
python crop_yield_prediction.py
```

All output and 10 plots will be generated automatically.

---

## 🔧 Technologies Used

- **Python 3**
- **Pandas** – data manipulation
- **NumPy** – numerical operations
- **Matplotlib & Seaborn** – data visualization
- **Scikit-learn** – machine learning models

---

## 📈 Algorithms Used

### 1. Random Forest Regression
- **Type**: Ensemble learning (Bagging)
- Builds **100 decision trees** in parallel
- Each tree trained on a random subset of data
- Final prediction = **average of all trees**
- Reduces overfitting by averaging multiple trees

### 2. Gradient Boosting Regression
- **Type**: Ensemble learning (Boosting)
- Builds trees **sequentially** (one after another)
- Each new tree **corrects the errors** of the previous tree
- Uses a **learning rate** to control each tree's contribution
- Often achieves slightly higher accuracy but needs more tuning

---

## 📏 Evaluation Metrics

| Metric | What it tells us |
|--------|-----------------|
| **MAE** (Mean Absolute Error) | Average of absolute differences between actual & predicted |
| **RMSE** (Root Mean Squared Error) | Similar to MAE but penalizes large errors more |
| **R² Score** | How well the model explains the variation (1.0 = perfect) |

---

## 📊 Project Steps (Flow)

1. Import Libraries
2. Load Dataset (Kaggle CSV)
3. Exploratory Data Analysis (EDA)
4. Data Visualization (6 plots)
5. Data Preprocessing (Label Encoding)
6. Train-Test Split (80/20)
7. Train Random Forest model
8. Train Gradient Boosting model
9. Compare both models
10. Actual vs Predicted plots
11. Feature Importance analysis
12. Predict on new data (demo)
13. Residual (error) analysis

---

## 👨‍🎓 Author
- **Name**: [Your Name]
- **Course**: Machine Learning – Semester 2
- **Year**: 2026
