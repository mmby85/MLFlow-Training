
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import fetch_california_housing

if __name__ == "__main__":
    print("extracting arguments")
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--n-estimators", type=int, default=100) # Increased default
    parser.add_argument("--min-samples-leaf", type=int, default=2) # Increased default

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN")) # No longer used
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST")) # No longer used
    parser.add_argument("--train-file", type=str, default="california-housing-train.csv") # Changed filename
    parser.add_argument("--test-file", type=str, default="california-housing-test.csv") # Changed filename

    args, _ = parser.parse_known_args()
    
    print("loading and preparing data")

    # Load the dataset directly.
    housing = fetch_california_housing(as_frame=True)
    housing_df = pd.DataFrame(housing.frame)

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(housing_df, test_size=0.2, random_state=42) # added train test split.

    print("saving training and testing datasets")

    # Save the training and testing data as CSVs in respective channels.
    # train_path = os.path.join(args.train, args.train_file)
    # test_path = os.path.join(args.test, args.test_file)
    train_df.to_csv("california-housing-train.csv", index=False)
    test_df.to_csv("california-housing-test.csv", index=False)

    # print(f"training data persisted at {train_path}")
    # print(f"test data persisted at {test_path}")

    print("building training and testing datasets")

    # Prepare the training and testing datasets
    X_train = train_df.drop("MedHouseVal", axis=1)
    X_test = test_df.drop("MedHouseVal", axis=1)
    y_train = train_df[["MedHouseVal"]]
    y_test = test_df[["MedHouseVal"]]

    # Train model
    print("training model")

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train.values.ravel()) # using .values.ravel() to get the 1d array

    # Print MSE
    print("validating model")

    mse_train = mean_squared_error(y_train, model.predict(X_train))
    mse_test = mean_squared_error(y_test, model.predict(X_test))

    print(f"Train MSE: {mse_train:.3f}")
    print(f"Test MSE: {mse_test:.3f}")

    # Persist model
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print("model persisted at " + path)
