# Wine-Quality-Prediction-Using-Machine-Learning-Techniques

#  Wine Quality Prediction 

This project performs a comprehensive analysis and classification of wine quality using various machine learning models. The dataset includes both **red** and **white** wine samples, and multiple preprocessing, visualization, and modeling steps were carried out to derive insights and build predictive systems.

---

##  Dataset Summary

- The dataset has **4898 samples** and **13 features**, including `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`, `quality`, and a derived feature `wine_type`.
- There are **no missing values** in the dataset. (`df.isnull().sum()` confirms this.)

---

##  Preprocessing Steps

- Dataset combined from red and white wine CSVs.
- A new column `wine_type` was added to differentiate between red and white wine.
- Label encoding was used for classification (`good_quality = quality >= 7`).
- Features and targets were scaled using `StandardScaler`.

---

##  Exploratory Data Analysis

Visualizations included:
- **Histogram plots** for alcohol content across wine types.
- **Correlation heatmaps** to show feature relationships with quality.
- **Boxplots** to study feature distribution by wine type.
- **Pair plots** and count plots for deeper insights into feature influence.

Key insights:
- `alcohol` is positively correlated with wine quality.
- Red and white wines exhibit different value distributions for acidity and sugar.

---

##  Feature Engineering

- `wine_type` was encoded as binary.
- Target variable for classification defined as:
  - `0` for low/average quality (quality < 7)
  - `1` for good quality (quality ≥ 7)

---

##  Machine Learning Models Used

### Classification Models:
1. **Logistic Regression**
2. **Random Forest**
3. **Gradient Boosting**

### Evaluation:
- **Cross-validation (5-fold) accuracy**:
  - Logistic Regression: `~74.70% ± 0.59%`
  - Random Forest: `~82.79% ± 0.66%` 
  - Gradient Boosting: `~83.22% ± 0.72%` 

- Confusion matrices and classification reports were used to assess precision, recall, and F1-scores.
- **Random Forest and Gradient Boosting** both outperform Logistic Regression.

---

##  Hyperparameter Tuning

Grid search was applied to Random Forest:
```python
rf_param = {
    'n_estimators'     : [100, 300],
    'max_depth'        : [None, 15],
    'min_samples_leaf' : [1, 2]
}
```

- Best Parameters:
  - `n_estimators`: 300
  - `max_depth`: None
  - `min_samples_leaf`: 1
- Accuracy after tuning improved slightly over the default model.

---

## Final Business Insights

- Alcohol content is a significant predictor of good quality wine. Higher alcohol levels are generally associated with better taste.
- Red wines are generally more acidic; white wines contain more residual sugar.
- Models like Random Forest and Gradient Boosting can successfully classify good vs average wines with high accuracy (over 83%).
- Feature importance plots highlight the most influential features for predicting wine quality.

---

## Conclusion

- The project achieves high classification accuracy using ensemble learning methods.
- Feature importance analysis and model interpretability provide actionable insights for wine producers to optimize for quality.
- Potential future work includes:
  - Using regression models to predict exact wine quality scores.
  - Deployment of the model via web app for real-time predictions.
