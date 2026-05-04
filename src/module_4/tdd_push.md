# Module 4: TDD

## Technical Description Document - Push Notifications DS

## Context

We are building a new product to enable the sales team to target customers to be sent a push notification to nudge them to buy a certain product that has been previously selected by them.

This tool is divided into 3 parts:

- Frontend: developed by the FE team  
- Backend engineering: data processing and serving (Engineering team)  
- Data science: predictive model development (this document)

## Goal

Develop a machine learning model that, given a user and a product, predicts if the user would purchase it at that moment.

> Constraint: Only purchases of **at least 5 items** are considered.

- Delivery: **Proof of Concept (POC) in 1 week**

## Data

- Dataset: `feature_frame_20210304.csv`
- EDA phase is skipped (already done previously)

### Required preprocessing

- Filter orders to keep only those with **≥ 5 items**

## Approach

### Milestone 1: Exploration Phase

Steps:

1. Filter dataset (orders ≥ 5 items)
2. Train multiple models:
   - Linear models
   - Non-linear models
   - Different parametrizations and regularizations

### Expected outcome

- Report / notebook including:
  - What worked
  - What did not work
  - Why
- Final model selected for next phase

### Milestone 2: MVP Code

Generate production-ready code for deployment in an API.

#### 1. Training (fit) function

Responsibilities:

- Receive model parameters
- Load data (via modular function)
- Train model
- Save model to disk

Naming convention:

```
push_yyyy_mm_dd
```

Example:

```python
def handler_fit(event, _):
    model_parametrisation = event["model_parametrisation"]

    # your code here

    return {
        "statusCode": "200",
        "body": json.dumps({
            "model_path": [your_model_stored_path],
        }),
    }
```

#### 2. Inference (predict) function

Input format:

```json
{
  "users": {
    "user_id": {"feature 1": value, "feature 2": value},
    "user_id2": {"feature 1": value, "feature 2": value}
  }
}
```

Output format:

```json
{
  "prediction": {
    "user_id": prediction,
    "user_id2": prediction
  }
}
```

Base implementation:

```python
def handler_predict(event, _):
    data_to_predict = pd.DataFrame.from_dict(
        json.loads(event["users"])
    )

    # your code here

    return {
        "statusCode": "200",
        "body": json.dumps({
            "prediction": {dict_of_predictions}
        }),
    }
```

## Notes

- You are free to modify structure if needed  
- Always document changes  
- Provide default values where necessary  