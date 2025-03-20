import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns from sklearn.metrics import confusion_matrix
import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/mukeshjangid7877/mlflow-demo.mlflow")
dagshub.init(repo_owner='mukeshjangid7877', repo_name='mlflow-demo', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)


# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)  

# Define the parameters for the Random Forest model
max_depth = 1
n_estimators = 100

# apply mlflow

mlflow.set_experiment('iris-rf')

with mlflow.start_run():

    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)

    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # Create a confusion matrix plot
  
    cn = ConfusionMatrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    # Save the plot as an artifact
    plt.savefig("confusion_matrix.png")

    # mlflow code
    mlflow.log_artifact("confusion_matrix.png")

    mlflow.log_artifact(__file__)

    mlflow.sklearn.log_model(rf, "random forest")

    mlflow.set_tag('author','rahul')
    mlflow.set_tag('model','random forest')

    print('accuracy', accuracy)
