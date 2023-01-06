# Heart_Attack_Prediction
This project is designed to predict whether a person is likely to have a heart attack or not based on various risk factors such as age, gender, blood pressure, cholesterol, etc. The prediction is made using different machine learning algorithms, including logistic regression, decision tree, random forest, K-nearest neighbors (KNN), and neural network.

## Dependencies 

NumPy

Pandas

Matplotlib

Seaborn

scikit-learn

Plotly

TensorFlow




## How to Run 
Clone this repository

Install the dependencies listed above

Open the Jupyter Notebook heart_attack_prediction.ipynb

Run the cells in the notebook to train and test the different machine learning models
Data

The data for this project is taken from the Cleveland Clinic Foundation for Heart Disease. It contains 14 variables and 303 observations, with a binary response variable indicating whether a person has heart disease or not.


## Machine Learning Models 
The following machine learning models are used to predict heart attack:

Logistic Regression

Decision Tree

Random Forest

K-Nearest Neighbors (KNN)

Neural Network (using TensorFlow)

## The following steps are taken in this project:

Importing necessary libraries: The first step is to import all the necessary libraries that will be used in the project. These libraries include numpy, pandas, seaborn, matplotlib, sklearn, and tensorflow.

Splitting the dataset into training and testing sets: The dataset is split into a training set (80%) and a testing set (20%) using train_test_split from sklearn.model_selection.

Applying machine learning algorithms: Several machine learning algorithms are applied to the training set, including LogisticRegression, DecisionTreeClassifier, KNeighborsClassifier, RandomForestClassifier, MLPClassifier, and SVC.

Evaluating model performance: The performance of each model is evaluated using various metrics, including r2_score, mean_squared_error, accuracy_score, and others. The model with the highest performance is selected for further analysis.

Visualizing results: The results of the analysis are visualized using various plots and charts, including confusion matrices, precision-recall curves, and ROC curves.

Improving model performance: Steps are taken to improve the performance of the selected model, such as hyperparameter tuning and feature selection.

Throughout the project, the focus is on understanding the underlying patterns in the data and using them to make accurate predictions.

