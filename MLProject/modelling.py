import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

mlflow.set_experiment("Insurance Regression")
mlflow.autolog()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "insurance_preprocessing.csv")

df = pd.read_csv(DATA_PATH)

X = df.drop("charges", axis=1)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42)
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mlflow.log_param("model_type", name)
        mlflow.log_metric("MAE", mean_absolute_error(y_test, preds))
        mlflow.log_metric("RMSE", np.sqrt(mean_squared_error(y_test, preds)))
        mlflow.log_metric("R2", r2_score(y_test, preds))
        
        mlflow.sklearn.log_model(model, "model")