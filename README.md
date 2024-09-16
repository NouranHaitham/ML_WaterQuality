# ML_WaterQuality 🚰

## **Project Overview**
Clean, safe drinking water is super important for health and development! 🌍💧 This project is all about predicting if water samples are potable (safe to drink) or not, using various physical and chemical properties. We’re diving into different machine learning models to figure out whether your water is ready for a sip or needs some more testing. 🧪📊

## **Objective**
We want to build a trusty classification model that can tell if water samples are safe to drink based on their features. We’ll use data preprocessing, exploratory data analysis (EDA), feature engineering, and machine learning algorithms to get the best results. 🏆💡

---

## **Project Structure**

### 1. **Data Preprocessing** 🛠️
   - **Handling Duplicates**: Double-checked and made sure there are no duplicate rows.
   - **Missing Values**: Filled in missing values for `ph`, `Sulfate`, and `Trihalomethanes` with mean and median values.
   - **Outlier Detection and Handling**: Found outliers, visualized them with box plots, and transformed those tricky features.
   - **Feature Scaling**: Scaled features like `Conductivity`, `Solids`, and `Hardness` using `MinMaxScaler` to help the models work better.
   - **Correlation Analysis**: Made a cool correlation heatmap to see how features relate to each other.

### 2. **Class Imbalance** ⚖️
   - Our dataset had more non-potable samples, so we used **SMOTE (Synthetic Minority Over-sampling Technique)** to balance things out. 

### 3. **Modeling** 🤖
   We tried out and compared several machine learning models:
   - **K-Nearest Neighbors (KNN)**: Tuned with `GridSearchCV` and checked accuracy, precision, recall, and F1-score.
   - **Decision Tree**: Adjusted parameters like `min_samples_split` and `max_depth` for better results.
   - **Random Forest**: Fine-tuned `n_estimators`, `min_samples_split`, and `max_depth` to boost accuracy.
   - **Other Models**: Also gave Logistic Regression and SVM (Support Vector Machine) a spin.

### 4. **Evaluation** 📈
   - We checked out models based on accuracy, precision, recall, F1-score, and confusion matrices.
   - Used **cross-validation** to make sure our models were solid and not overfitting.
   - **Best Model**: KNN with `n_neighbors = 23` was the champ based on cross-validation and overall performance.

---

## **Dataset**

The dataset is publicly available and includes:
- **pH**: pH level of the water (0 to 14).
- **Hardness**: How hard the water is, measured in mg/L.
- **Solids**: Total dissolved solids in ppm.
- **Chloramines**: Amount of chloramines in ppm.
- **Sulfate**: Amount of sulfates in mg/L.
- **Conductivity**: Electrical conductivity in μS/cm.
- **Organic Carbon**: Amount of organic carbon in ppm.
- **Trihalomethanes**: Amount of Trihalomethanes in μg/L.
- **Turbidity**: How clear the water is, measured in NTU.
- **Potability**: Whether the water is potable (1) or not (0).

---

## **Installation** 🛠️

1. Clone the repo:
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
## **Usage** 🚀

1. **Exploratory Data Analysis (EDA)** 🔍:
   Run the `data_analysis.ipynb` notebook to dive into the dataset, spot missing values, and explore how features connect to water potability.

2. **Data Preprocessing** 🛠️:
   The `preprocessing.py` script handles missing values, detects outliers, and scales features. Feel free to tweak it if you need extra steps!

3. **Model Training** 🤖:
   Use `train_models.py` to train several machine learning models, including KNN, Decision Trees, and Random Forests. You can adjust hyperparameters directly in the script.

4. **Model Evaluation** 📊:
   After training, the script will show you how each model performs with metrics like accuracy, precision, recall, F1-score, and confusion matrix. Perfect for comparing results!

## **Technologies Used** 💻

- **Python**: Our main coding language.
- **Pandas**: For all your data manipulation needs.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For machine learning magic.
- **Matplotlib & Seaborn**: For awesome data visualizations.
- **SMOTE**: To balance out the dataset.
- **GridSearchCV**: For fine-tuning hyperparameters and getting the best performance.

## **Results** 🌟

The star of the show is the KNN model with `n_neighbors = 23`, achieving:
- **Accuracy**: 79.5%
- **Precision**: 81.3%
- **Recall**: 76.9%
- **F1-Score**: 79.0%

## **Future Improvements** 🔧

- Experiment with more models like XGBoost or LightGBM.
- Implement feature selection to streamline the model.
- Further tune hyperparameters with RandomizedSearchCV.
- Deploy the model as an API to make real-time water potability predictions.
