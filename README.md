# ML_WaterQuality

## **Project Overview**
Access to clean, safe drinking water is essential for human health and development. This project focuses on predicting the potability of water samples based on various physical and chemical properties. We explore different machine learning models to assess whether water is potable (safe for human consumption) or non-potable using a dataset of water quality features such as pH, hardness, solids, chloramines, sulfate, and more.

## **Objective**
The goal of this project is to build a reliable classification model that can predict the potability of water samples, ensuring that water safety is accurately assessed based on various input features. This project applies data preprocessing, exploratory data analysis (EDA), feature engineering, and machine learning algorithms to achieve optimal results.

---

## **Project Structure**

### 1. **Data Preprocessing**
   - **Handling Duplicates**: Checked and confirmed no duplicate rows in the dataset.
   - **Missing Values**: Imputed missing values for `ph`, `Sulfate`, and `Trihalomethanes` using a combination of mean and median imputation based on potability.
   - **Outlier Detection and Handling**: Detected outliers in numerical columns, visualized using box plots, and performed transformation on outlier-prone features.
   - **Feature Scaling**: Normalized numerical features like `Conductivity`, `Solids`, `Hardness`, etc., using `MinMaxScaler` for better model convergence.
   - **Correlation Analysis**: Generated a correlation heatmap to assess multicollinearity and feature relevance.

### 2. **Class Imbalance**
   - The dataset was imbalanced, so we applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the class distribution between potable and non-potable water samples.

### 3. **Modeling**
   We implemented and compared several machine learning models:
   - **K-Nearest Neighbors (KNN)**: Tuned using `GridSearchCV` and evaluated based on accuracy, precision, recall, and F1-score.
   - **Decision Tree**: Hyperparameters such as `min_samples_split` and `max_depth` were tuned to optimize model performance.
   - **Random Forest**: Tuned key hyperparameters like `n_estimators`, `min_samples_split`, and `max_depth` for improved accuracy.
   - **Other Models**: Additional models such as Logistic Regression and SVM (Support Vector Machine) were tested.

### 4. **Evaluation**
   - Models were evaluated based on accuracy, precision, recall, F1-score, and confusion matrices.
   - **Cross-validation** was used to ensure model robustness and prevent overfitting.
   - **Best Model**: The KNN model with `n_neighbors = 23` was selected based on its cross-validation results and overall performance on the test set.

---

## **Dataset**

The dataset used in this project is publicly available and includes the following features:
- **pH**: pH value of water (0 to 14).
- **Hardness**: Capacity of water to precipitate soap, measured in mg/L.
- **Solids**: Total dissolved solids in ppm.
- **Chloramines**: Amount of chloramines in ppm.
- **Sulfate**: Amount of sulfates dissolved in mg/L.
- **Conductivity**: Electrical conductivity of water in μS/cm.
- **Organic Carbon**: Amount of organic carbon in ppm.
- **Trihalomethanes**: Amount of Trihalomethanes in μg/L.
- **Turbidity**: Measure of water's light-emitting property in NTU.
- **Potability**: Binary classification (1 = potable, 0 = non-potable).

---

## **Installation**

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/water-potability-prediction.git
   ```
   
2. Navigate to the project directory:
   ```bash
   cd water-potability-prediction
   ```
   
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
4. Download the dataset and place it in the root directory:

 - Water Potability Dataset

Place the dataset in the root directory of the project after downloading.

5. Run the Project Notebooks:
For Exploratory Data Analysis (EDA) and Data Preprocessing, navigate to the notebooks folder and run the Jupyter notebooks:
```bash
jupyter notebook data_analysis.ipynb
```
6. Run the Model Training:
After preprocessing the data, you can train the models by executing the train_models.py script:
```bash
python train_models.py
```
## **Usage**

1. **Exploratory Data Analysis (EDA)**:
   Run the `data_analysis.ipynb` notebook to visualize the dataset, check for missing values, and analyze the relationships between features and the target variable (potability).

2. **Data Preprocessing**:
   The `preprocessing.py` script handles missing value imputation, outlier detection, and feature scaling. You can modify the script for additional preprocessing steps if needed.

3. **Model Training**:
   Use `train_models.py` to train multiple machine learning models, including K-Nearest Neighbors (KNN), Decision Trees, and Random Forests. You can specify hyperparameter tuning options in the script.

4. **Model Evaluation**:
   After training the models, the script outputs accuracy, precision, recall, F1-score, and confusion matrix for each model to help you evaluate their performance.

## **Technologies Used**

- **Python**: Core programming language.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical operations.
- **Scikit-learn**: Machine learning models and utilities.
- **Matplotlib & Seaborn**: Data visualization.
- **SMOTE**: Balancing the dataset.
- **GridSearchCV**: Hyperparameter tuning for optimal model performance.

## **Results**

The best performing model (KNN with n_neighbors = 23) achieved the following metrics on the test set:
- **Accuracy**: 79.5%
- **Precision**: 81.3%
- **Recall**: 76.9%
- **F1-Score**: 79.0%

## **Future Improvements**

- Explore additional models such as XGBoost or LightGBM.
- Implement feature selection techniques to reduce dimensionality.
- Further tuning of hyperparameters using RandomizedSearchCV.
- Deployment of the model as an API for real-time water potability predictions.


