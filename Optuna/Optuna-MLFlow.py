import mlflow
import optuna
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np
from time import time
from mlflow.models.signature import infer_signature
from optuna.samplers import TPESampler
from optuna.integration import MLflowCallback
from sklearn.base import BaseEstimator


mlflow.set_tracking_uri("http://127.0.0.1:5000")


# Load the California Housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define a function to evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, r2, mae


MODEL_REGISTRY = {
    "RandomForest": {
        "class": RandomForestRegressor,
        "params": {
            "n_estimators": (lambda trial: trial.suggest_int("n_estimators", 100, 500)),
            "max_depth": (lambda trial: trial.suggest_int("max_depth", 5, 15)),
            "min_samples_split": (lambda trial: trial.suggest_int("min_samples_split", 2, 10)),
            "min_samples_leaf": (lambda trial: trial.suggest_int("min_samples_leaf", 1, 5)),
            "random_state": 42
        },
    },
    "GradientBoosting": {
        "class": GradientBoostingRegressor,
        "params": {
            "n_estimators": (lambda trial: trial.suggest_int("n_estimators", 100, 500)),
            "max_depth": (lambda trial: trial.suggest_int("max_depth", 3, 10)),
            "learning_rate": (lambda trial: trial.suggest_float("learning_rate", 0.01, 0.2, log=True)),
            "random_state": 42
        },
    },
    "LinearRegression": {"class": LinearRegression, "params": {}},
}


def define_model(trial, model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model name: {model_name}")

    model_config = MODEL_REGISTRY[model_name]
    params = {
        name: param(trial) if callable(param) else param
        for name, param in model_config["params"].items()
    }
    model = model_config["class"](**params)

    return model, params


def objective(trial):
    model_name = trial.suggest_categorical("model", list(MODEL_REGISTRY.keys()))
    run_name = f"Trial_{trial.number}_{model_name}"

    with mlflow.start_run(nested=True, run_name=run_name) as nested_run:
        mlflow.set_tag("model_type", model_name)
        mlflow.log_param("trial_number", trial.number)

        model, params = define_model(trial, model_name)

        mlflow.log_params(params)

        start_time = time()
        model.fit(X_train, y_train)
        end_time = time()
        training_time = end_time - start_time
        mlflow.log_metric("training_time", training_time)

        mse, r2, mae = evaluate_model(model, X_test, y_test)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

    return mse


def early_stopping_callback(study, trial):
    if trial.number > 5 and len(study.trials) > 20:
        current_best_mse = study.best_value
        current_trials = [trial_i.value for trial_i in study.trials if trial_i.value is not None]
        if len(current_trials) > 10:
            median_value = np.median(current_trials)
            if current_best_mse > median_value * 1.2:
                study.stop()
                print(f"Early stopped trial {trial.number}")


def retrain_best_model(best_trial, X_train, y_train):
    best_model_type = best_trial.params['model']
    best_model, _ = define_model(best_trial, best_model_type)
    best_model.fit(X_train, y_train)
    return best_model


if __name__ == "__main__":
    mlflow.set_experiment("optuna_mlflow_housing_advanced")
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    with mlflow.start_run(run_name="Optuna_Study") as run:
        mlflow.log_input(mlflow.data.from_pandas(X), "features")
        mlflow.log_input(mlflow.data.from_pandas(pd.DataFrame(y), source="target"), "target")

        mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), mlflow_kwargs={"nested": True})
        study.optimize(objective, n_trials=50, callbacks=[early_stopping_callback, mlflow_callback])

        best_trial = study.best_trial

        mlflow.log_params({f"best_{k}": v for k, v in best_trial.params.items()})
        mlflow.log_metric("best_mse", best_trial.value)

        best_model = retrain_best_model(best_trial, X_train, y_train)

        signature = infer_signature(X_test, best_model.predict(X_test))
        mlflow.sklearn.log_model(best_model, artifact_path="best_model", signature=signature)

        mse, r2, mae = evaluate_model(best_model, X_test, y_test)
        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_r2", r2)
        mlflow.log_metric("test_mae", mae)

        print(f"Best trial value (MSE): {best_trial.value}")
        print(f"Best parameters: {best_trial.params}")