# ASG | Titanic Survival Prediction

This repository contains a complete, end-to-end machine learning project that predicts passenger survival based on the classic Titanic dataset from Kaggle.
This project serves as a fundamental exercise in building a classification model, covering the entire workflow from initial data exploration and preprocessing to model training and performance evaluation.

## Prediction Process
The process in the notebook follows:
    
    - Exploratory Data Analysis (EDA) & Preprocessing:
        Loaded the dataset and performed an initial inspection.
        Handled missing values for Age (imputed with median) and Embarked (imputed with mode).
        Removed irrelevant columns (Cabin, PassengerId, Name, Ticket).
        Performed Label Encoding on the Sex column and Normalization on the Age column to prepare the data for modeling.
        
    - Model Training:
        The preprocessed data was split into training (80%) and testing (20%) sets.
        A Logistic Regression model was trained on the training data to learn the patterns associated with passenger survival.
        
    - Model Evaluation:
        The model's performance was evaluated on the unseen test data.
        Key metrics such as Accuracy, Precision, Recall, and F1-Score were calculated.
        A Confusion Matrix and ROC Curve were generated to visually assess the model's classification performance.
## Results
The trained Logistic Regression model achieved the following performance on the test set:
    
    Accuracy: ~79% (The model correctly predicted the survival outcome for about 79% of the passengers in the test data).
    Precision (for survived class): 77%
    Recall (for survived class): 69%
Meanwhile, the Naive Bayes model achieved the following performance on the test set:
    
    Accuracy: ~77% (The model correctly predicted the survival outcome for about 77% of the passengers in the test data).
    Precision (for survived class): 73%
    Recall (for survived class): 72%

The confusion matrix provides a detailed breakdown of correct and incorrect predictions for both classes (survived and not survived).

## Tools Used
- Python
- Pandas: for data manipulation and analysis.
- Matplotlib & Seaborn: for data visualization.
- Scikit-learn: for data preprocessing (LabelEncoder, MinMaxScaler).
- Jupyter Notebook as the working environment.
