# Credit Risk Modeling & AI Decision Support Platform

End-to-end credit risk modeling and AI decision support platform built around the Home Credit Credit Risk Model Stability dataset.

## Overview

This project is designed as a production-oriented credit risk platform rather than a standalone modeling notebook. It combines relational data engineering, supervised and unsupervised machine learning, neural networks, hyperparameter optimization, explainability, temporal stability analysis, MLOps, API deployment, cloud integration, and business-facing decision support.

## Business Problem

The core objective is to estimate borrower default risk while emphasizing not only predictive performance, but also model calibration, temporal stability, explainability, operational reliability, and decision-support usefulness.

The platform will evaluate multiple modeling approaches and select a champion model based on a broader risk framework rather than a single aggregate metric.

## Planned Architecture

- Home Credit Credit Risk Model Stability data
- Python, Polars, PyArrow, SQL
- PostgreSQL relational data platform
- Reproducible data validation and feature pipelines
- Logistic Regression baseline
- Random Forest
- LightGBM
- XGBoost
- CatBoost
- MLP neural network
- Temporal / sequence neural-network experiments
- Unsupervised segmentation and anomaly detection
- Optuna hyperparameter optimization
- Calibration and model stability analysis
- SHAP and model explainability
- MLflow experiment tracking
- FastAPI scoring service
- Docker
- GitHub Actions
- Tableau decision-support dashboards
- Azure deployment demonstrations
- Controlled AI-generated explanations

## Modeling Strategy

The project will compare multiple model families under a consistent temporal validation framework.

Evaluation will include:

- ROC-AUC
- Gini coefficient
- PR-AUC
- KS statistic
- Brier score
- Calibration
- Lift and gain
- Risk-decile analysis
- Temporal stability
- Population Stability Index (PSI)
- Subgroup diagnostics
- Threshold and policy analysis

The final production candidate will be selected based on predictive performance, calibration, stability, explainability, and operational suitability.

## Data Source

Primary dataset:

**Home Credit – Credit Risk Model Stability (2024)**

The source data contains application-level information and multiple related historical data tables with different relational depths.

Raw source files will not be committed to Git. Data acquisition and ingestion will be reproducible through project scripts and documentation.

## Repository Structure

```text
config/          Project configuration
data/            Local raw, interim, processed, and external data
docs/            Architecture, data, modeling, and deployment documentation
infrastructure/  Cloud and deployment configuration
notebooks/       Exploratory analysis and controlled experiments
scripts/         Project utility and execution scripts
sql/             PostgreSQL DDL, staging, feature, and analytics SQL
src/             Production Python package
tableau/         Tableau-related project assets
tests/           Unit and integration tests
artifacts/       Local model and experiment artifacts
```

## Development Principles

- Reproducible and testable pipelines
- Temporal validation over random-only validation
- Model complexity must be justified empirically
- No model is assumed to outperform before evaluation
- Neural-network and unsupervised methods are included as genuine analytical experiments
- AI-generated explanations do not directly determine credit decisions
- Cloud components must satisfy the project's $0 total-cost constraint
- Technologies are documented as implemented, not merely planned

## Status

Initial development

**Current phase:** repository foundation and data-platform setup.