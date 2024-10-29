
# Telecom Customer Churn Prediction

## Project Overview
This project aims to analyze and predict **customer churn** for a telecommunications company operating in California. Using a variety of machine learning models, the goal is to understand the key indicators behind customer churn, enabling the company to take proactive measures to retain at-risk customers. The analysis also explores the influence of customer demographics, billing details, and service usage patterns on churn behavior.

---

## Table of Contents
1. [Dataset](#dataset)
2. [Project Structure](#project-structure)
3. [Data Preprocessing](#data-preprocessing)
4. [Visualization](#visualization)
5. [Machine Learning Models](#machine-learning-models)
6. [Performance Evaluation](#performance-evaluation)
7. [How to Run](#how-to-run)
8. [References](#references)

---

## Dataset
- **Source**: [Kaggle - Telecom Customer Churn by Maven Analytics](https://www.kaggle.com/datasets/shilongzhuang/telecom-customer-churn-by-maven-analytics/data)
- **Rows**: 7043  
- **Columns**: 38  
- Contains customer demographic information, subscribed services, billing details, and reasons for churn.

---

## Project Structure
```
├── DME_Project_Final.ipynb  # Jupyter Notebook with full analysis and model implementation
├── DME Project Final.pdf    # Project report detailing findings and insights
├── README.md                # This README file
```

---

## Data Preprocessing
- **Handling Missing Values**: Imputed missing values using the average or most frequent values in categorical columns.
- **Categorical Encoding**: Converted Yes/No columns to 1/0 using label encoding.
- **Duplicates**: Verified there are no duplicate records in the dataset.

---

## Visualization
The project includes interactive visualizations to gain insights into customer behavior and churn patterns:
1. **Churn Reasons**: [Interactive visualization](https://public.flourish.studio/visualisation/16822273/)
2. **Customer Status by Gender**: [Interactive visualization](https://public.flourish.studio/visualisation/16832837/)
3. **Revenue Analysis**: [Interactive visualization](https://public.flourish.studio/visualisation/16832669/)
4. **Service Usage**: [Interactive visualization](https://public.flourish.studio/visualisation/16833063/)
5. **Age Group Analysis**: Bar chart visualizing age-based segmentation.
6. **Geographical Distribution**: Map showing customer density across California.

---

## Machine Learning Models
The following models were used to predict churn:
1. **Logistic Regression**  
2. **Decision Trees**  
3. **Random Forest**  
4. **Support Vector Machines (SVM)**  
5. **Gradient Boosting Machines (GBM)**  
6. **Neural Networks**  
7. **XGBoost and LightGBM**

**Chosen Model**:  
- **Random Forest** provided the best accuracy of 92.87% and was selected for the final prediction model.

---

## Performance Evaluation
- **ROC-AUC Score**: Random Forest achieved the highest AUC of 0.92, demonstrating strong predictive performance.
- All models exhibited consistent performance, but Random Forest showed slightly better generalization.

---

## How to Run
1. **Install Dependencies**:  
   Ensure you have the following Python libraries installed:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```
2. **Open the Notebook**:
   ```bash
   jupyter notebook DME_Project_Final.ipynb
   ```
3. **Run All Cells**:
   - Execute the cells sequentially to see the data preprocessing, visualization, and model training steps.

---

## References
- Kaggle Dataset: [Customer Churn Data](https://www.kaggle.com/datasets/shilongzhuang/telecom-customer-churn-by-maven-analytics/data)  
- Exploratory Data Analysis in Python: [EDA Guide](https://towardsdatascience.com/exploratory-dataanalysis-eda-python-87178e35b14)  
- Label Encoding Techniques: [Guide](https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd)

---

## Contributors
- **Gaurav Ramoliya**  
- **Parth Deshmukh**  
- **Malav Makadia**

---

This project provides insights into customer churn dynamics and demonstrates the application of machine learning techniques in real-world scenarios. The Random Forest model was chosen for deployment due to its superior performance in predicting churn.
