# Fuel-Cell-Performance-Assignment

This project focuses on comparing various machine learning models and preprocessing techniques to optimize performance metrics using the **PyCaret** library. The dataset used is related to **Fuel Cell Performance**, with the goal of predicting the `Target5` variable.

## Authors

- [@Guryansh](https://www.github.com/Guryansh)

  
## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
5. [Results](#results)

## Project Overview

This project employs the **PyCaret Regression** module to analyze different preprocessing techniques and modeling approaches, such as:
- **Normalization**
- **Feature Selection**
- **Outlier Removal**
- **Transformation**
- **Principal Component Analysis (PCA)**

Models are evaluated under various conditions, including 10-fold and 15-fold cross-validation. The best-performing model is further optimized using advanced configurations.

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Guryansh/Fuel-Cell-Performance-Assignment
    ```
2. Install required dependencies:
    ```bash
    pip install pycaret pandas
    ```

3. Ensure the dataset `Fuel_cell_performance_data-Full.csv` is in the same directory as the script.

---

## Dataset

The dataset contains several performance-related features of fuel cells. The target variable is `Target5`. The data is preprocessed to select relevant features and split into training and test sets.

---

## Methodology

The project explores the following preprocessing techniques:
- **Normalization**: Adjusting data to follow a standard distribution (z-score method).
- **Feature Selection**: Selecting the most relevant features for prediction.
- **Outlier Removal**: Using the Isolation Forest method to remove extreme values.
- **Transformation**: Applying Yeo-Johnson transformation for data normalization.
- **PCA**: Reducing dimensionality for improved model performance.

Each preprocessing step is evaluated under two configurations:
- **10-Fold Cross-Validation**
- **15-Fold Cross-Validation**

The top-performing model is determined for each configuration and recorded with its R² score.

---

## Results

The results are stored in a `model_df` DataFrame, summarizing:

| Model Code                  | Model Parameters             | R²     |
|-----------------------------|------------------------------|--------|
| ExtraTreesRegressor         | None, 10 folds              | 0.7711 |
| ExtraTreesRegressor         | None - shuffle, 10 folds    | 0.7674 |
| LGBMRegressor               | None, 15 folds              | 0.7676 |
| GradientBoostingRegressor   | None, 15 folds              | 0.7543 |
| ExtraTreesRegressor         | Normalization, 10 folds     | 0.7677 |
| ExtraTreesRegressor         | Normalization, 15 folds     | 0.7732 |
| ExtraTreesRegressor         | Feature Selection, 10 folds | 0.6909 |
| ExtraTreesRegressor         | Feature Selection, 15 folds | 0.6936 |
| LGBMRegressor               | Outlier Removal, 10 folds   | 0.7671 |
| LGBMRegressor               | Outlier Removal, 15 folds   | 0.7621 |
| ExtraTreesRegressor         | Transformation, 10 folds    | 0.7704 |
| LGBMRegressor               | Transformation, 15 folds    | 0.7684 |
| LGBMRegressor               | PCA, 10 folds               | 0.7768 |
| LGBMRegressor               | PCA, 15 folds               | 0.7708 |
| ExtraTreesRegressor         | N+OR, 10 folds              | 0.7606 |
| LGBMRegressor               | N+T, 10 folds               | 0.7680 |
| BayesianRidge               | N+PCA, 10 folds             | 0.7140 |
| ExtraTreesRegressor         | OR+PCA, 10 folds            | 0.7616 |
| LGBMRegressor               | OR+T, 10 folds              | 0.7619 |
| LGBMRegressor               | PCA+T, 10 folds             | 0.7696 |
| AdaBoostRegressor           | PCA - kernel, 10 folds      | -0.0099|
| ExtraTreesRegressor         | PCA - incremental, 10 folds | 0.7680 |

A combination of **PCA** and other techniques provided the best results. Further optimization was performed using PCA methods such as kernel-based and incremental PCA.

Best Case - Light Gradient Boosting Machine Regressor, PCA (linear) -> 0.7768

Worst Case - AdaBoost Regressor, PCA (kernel) -> -0.0099

---

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
