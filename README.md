# Medical Resource Allocation using Machine Learning

## Overview
This project implements a machine learning pipeline to simulate patient prioritization during emergency medical scenarios. The goal is to classify patients based on urgency using vital health indicators such as heart rate, oxygen levels, and age.

## Problem Statement
In emergency healthcare systems, timely and accurate patient prioritization is critical. This project models a triage system using machine learning to assist decision-making and reduce delays in critical care.

## Approach
- Data preprocessing using median imputation to handle missing medical values  
- Stratified train-test split to maintain class balance  
- Random Forest classifier to capture non-linear relationships in patient vitals  
- Hyperparameter tuning (max_depth, min_samples_leaf) to balance bias-variance tradeoff  
- Model evaluation using Precision, Recall, and F1-score  

## Model Details
- Algorithm: Random Forest Classifier  
- n_estimators: 100  
- max_depth: 10  
- min_samples_leaf: 2  
- random_state: 42  

## Key Features
- Handles missing data robustly  
- Captures complex interactions between patient vitals  
- Provides interpretable predictions suitable for healthcare scenarios  

## Results
The model was evaluated using F1-score to balance precision and recall.  
Special focus was given to minimizing false negatives (critical patients misclassified as non-critical).

## Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  

## Project Structure
medical-resource-allocation-ml/
│── train.py  
│── data.csv  
│── requirements.txt  
│── README.md  

## How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Run the model:
python train.py

## Future Improvements
- Integration with real-time hospital data systems  
- Deployment using FastAPI  
- Use deep learning for advanced diagnosis  

## Author
Libiya Mariya Jimmy
