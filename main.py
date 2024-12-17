import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes


# Load data
data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train a simple model
n_estimators=15
random_state=32

model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
model.fit(X_train, y_train)

# Make predictions and calculate metrics
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)


#important !!
mlflow.set_tracking_uri('http://localhost:5000')


# Start an MLflow run
with mlflow.start_run():

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    # Make predictions and calculate metrics
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")
    print(f"Model logged with MSE: {mse}")



experiment_name = "Diabetes"
mlflow.set_experiment(experiment_name)

    
with mlflow.start_run(run_name=f"diabetes_RForest_{n_estimators}"):

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    # Make predictions and calculate metrics
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")
    print(f"Model logged with MSE: {mse}")

experiment_name = "Diabetes"
mlflow.set_experiment(experiment_name)

# mlflow.set_tracking_uri('http://13.61.136.59:5000/')

random_state=500

for n_estimators in [50, 55]:
    
    with mlflow.start_run(run_name=f"diabetes_RForest_{n_estimators}"):

        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)

        # Make predictions and calculate metrics
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "model")
        print(f"Model logged with MSE: {mse}")

        # Log model to MLflow
        input_example = pd.DataFrame(data['data'], columns=data['feature_names']).sample(1)
        mlflow.sklearn.log_model(sk_model=model, artifact_path='model',input_example=input_example)
