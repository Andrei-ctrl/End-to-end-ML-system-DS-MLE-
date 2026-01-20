# End-to-end ML system DS-MLE
This project demonstrates how a machine learning model is transformed into a production-ready system, covering the full lifecycle from training to deployment and monitoring.

# End-to-End ML System (DS + MLE)

This project demonstrates a **production-style end-to-end machine learning system**
covering the full ML lifecycle:

**training → model artifacts → inference API → containerization → monitoring → data drift → retraining**

The goal is to showcase the skills that distinguish an **ML Engineer** from a pure Data Scientist.

## Problem Statement

Predict the probability of **user churn / non-retention** based on tabular user activity data.

**Task:** Binary classification  
**Metric:** ROC-AUC, Precision/Recall  
**Model:** Scikit-learn classifier  
**Serving:** FastAPI  

---

## System Architecture

```text
            ┌──────────────┐
            │  Raw Data    │
            └──────┬───────┘
                   │
           Training Pipeline
                   │
        ┌──────────▼──────────┐
        │  Model + Features   │
        │  Artifacts          │
        └──────────┬──────────┘
                   │
          ┌────────▼─────────┐
          │  FastAPI Service │
          │  /predict        │
          └────────┬─────────┘
                   │
     ┌─────────────▼─────────────┐
     │ Logs • Metrics • Drift    │
     └─────────────┬─────────────┘
                   │
              Retraining
