from sklearn.datasets import load_iris
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from urllib.parse import urlparse

from sklearn.linear_model import LogisticRegression  #Logistic Regression
from sklearn.model_selection import train_test_split #to split the dataset for training and testing
from sklearn.neighbors import KNeighborsClassifier  #K nearest neighbours
from sklearn import svm  #Support Vector Machine
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn import preprocessing

import os
import warnings
import sys



import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the iris data
    data = load_iris()    
    try:
        df = pd.DataFrame(data= np.c_[data['data'], data['target']],
                        columns= data['feature_names'] + ['target'])
        df.columns = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']
    
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e)

    # Change columns name and define features and label 
    Flower_feature=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    y=df['Species']
    X=df[Flower_feature]
    
    # Train test split 
    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=.8, random_state = 0)
    
    # Standard scaler data 
    X_train_scaled = preprocessing.scale(train_X)
    X_test_scaled = preprocessing.scale(test_X)
    
    max_iter = float(sys.argv[1]) if len(sys.argv) > 1 else 100
    
    # ML Flow Start
    with mlflow.start_run():
        
        # Create model, train it
        model = LogisticRegression(max_iter = max_iter)
        model.fit(X_train_scaled,train_y)

        # Model Prediction
        prediction=model.predict(X_test_scaled)
       
        # Log model
        mlflow.sklearn.log_model(model, "model-lr")
        
        # Log params
        print("Elasticnet model (max_iter=%f):" % (max_iter))
        mlflow.log_param("max_iter", max_iter)

        # Create metrics
        acc = metrics.accuracy_score(prediction,test_y)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        print("Accuracy: %s" % acc)
       
        # Create feature importance


        # Log importances using a temporary file
        
  
        