# Heart_Attack_Prediction
This project is designed to predict whether a person is likely to have a heart attack or not based on various risk factors such as age, gender, blood pressure, cholesterol, etc. The prediction is made using different machine learning algorithms, including logistic regression, decision tree, random forest, K-nearest neighbors (KNN), and neural network.

## Dependencies 1

NumPy
Pandas
Matplotlib
Seaborn
scikit-learn
Plotly
TensorFlow

## How to Run 1
Clone this repository
Install the dependencies listed above
Open the Jupyter Notebook heart_attack_prediction.ipynb
Run the cells in the notebook to train and test the different machine learning models
Data
The data for this project is taken from the Cleveland Clinic Foundation for Heart Disease. It contains 14 variables and 303 observations, with a binary response variable indicating whether a person has heart disease or not.

## Machine Learning Models 1
The following machine learning models are used to predict heart attack:

Logistic Regression
Decision Tree
Random Forest
K-Nearest Neighbors (KNN)
Neural Network (using TensorFlow)
The models are trained and tested using a 80/20 train/test split. The performance of each model is evaluated using various metrics, including accuracy, precision, recall, and the area under the receiver operating characteristic (ROC) curve.

## Conclusion 1
Based on the results obtained, the neural network model appears to have the highest accuracy and the highest area under the ROC curve, followed by the random forest and KNN models. The logistic regression and decision tree models have relatively lower performance compared to the other models.
