import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import dagshub


# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)  

# Train a RandomForest model
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
