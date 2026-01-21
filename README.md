# End-to-End ML System with Drift Detection and Automated Retraining

This project demonstrates a production-oriented machine learning system for customer churn prediction, covering the full lifecycle from model training to monitoring, drift detection, automated retraining with safeguards, testing and containerization.


## Project Overview

Business problem
Predict customer churn and automatically react when the data distribution changes in production.

Key features

* End-to-end ML pipeline (training → inference → monitoring)
* Data drift detection using Evidently
* Automatic retraining triggered by drift
* Performance regression guard (model is replaced only if better)
* Cooldown mechanism to prevent retraining loops
* Unit tests for critical logic
* Dockerized for reproducibility


## Model Training

The model used in this project is a Logistic Regression classifier. Before training, the data is preprocessed by scaling numerical features with a standard scaler and encoding categorical features using one-hot encoding. The model is evaluated using ROC-AUC and accuracy to measure its performance. All experiments are tracked locally using MLflow so that metrics and parameters are reproducible.

After training several artifacts are saved. The trained model is stored as a serialized file, the evaluation metrics are saved in a JSON file and baseline statistics from the training data are stored to be used later for drift detection.


## Drift Detection

Data drift is detected using Evidently’s DataDriftPreset. The system compares the original training data, which serves as a reference, with new data collected during inference. This inference data is extracted from application logs.

The drift detection step produces an interactive HTML report for inspection and a JSON summary that is used programmatically. If a large enough share of features shows drift compared to the reference data, the system considers the model a candidate for retraining.


## Automated Retraining with Guard

When drift exceeds a predefined threshold, the system may retrain the model automatically. Before retraining starts, a cooldown check is applied to prevent retraining too often. The timestamp of the last retraining is stored in a state file inside the artifacts directory.

If retraining is allowed, a new model is trained on the updated data. After training, the new model is compared against the currently deployed model using ROC-AUC. The new model is accepted only if it performs at least as well as the existing one. This performance guard prevents the system from replacing a good model with a worse one and makes automated retraining safe.


## Testing

The project includes unit tests for the most important parts of the system. These tests verify the drift decision logic, ensure that the cooldown mechanism works correctly, confirm that inference logs are parsed properly and check that the retraining guard behaves as expected.

All tests can be run using pytest and they must pass before the system is considered ready.


## Docker and Reproducibility

The entire project is containerized using Docker. By running Docker Compose the full system can be built and executed in a clean and reproducible environment. This ensures consistent dependencies, avoids local environment issues and closely resembles how the system would run in production.


## Tech Stack

* Python 3.11
* pandas, NumPy, scikit-learn
* Evidently (data drift)
* MLflow (experiment tracking)
* pytest (testing)
* Docker & Docker Compose

## Future possible extensions

* CI/CD pipeline
* Model registry promotion stage
* Alerting (Slack / email)

