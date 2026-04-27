# Module 3: TDD — Technical Description Document
## Push Notifications — Data Science

---

## Context

We are building a new product to enable the sales team to target customers to be sent a push notification, nudging them to buy a product previously selected by the sales team.

This tool is divided into 3 parts:

| Area | Responsibility |
|---|---|
| **Frontend** | UI/UX — carried out by the FE team |
| **Backend Engineering** | Data availability and serving to the frontend — owned by the Engineering team |
| **Data Science** | Development of a predictive model — covered in this document |

---

## Goal

Develop a machine learning model that, given a **user** and a **product**, predicts whether the user would purchase it if they were shopping at that point in time.

- We should only focus on purchases of **at least 5 items**, as per the sales team requirement.
- The sales team expects a **Proof of Concept (PoC) within one week**.

---

## Data

We will be using the groceries dataset: **`feature_frame_20210304.csv`**

This is a well-known dataset from previous work over the last several weeks. For this reason, this project **skips the Exploratory Data Analysis (EDA) phase** — it should already be available in prior reports.

> **Required tweak:** filter orders to keep only those with **5 or more items**, as these are our target population.

---

## Approach

### Milestone 1: Exploration Phase

Given our existing understanding of the data, we jump directly into building the predictive model.

**Steps:**

1. **Filter data** — retain only orders with 5+ items to construct the working dataset.
2. **Model building** — to meet the tight PoC deadline, we limit ourselves to **linear models**, evaluated using a **train / validation / test split**.

**Expected outputs:**
- A report, notebook, or documentation covering what worked, what didn't, and why.
- A **final selected model** to carry forward into Milestone 2.

---

### Milestone 2: MVP Code

Using the outcomes of Milestone 1, production-ready MVP code must be generated and shared with the Engineering team for deployment.

The pipeline should include the following steps:

1. **Data loading** — loads data and applies validations where required.
2. **Pre-processing** — handles any required pre-processing steps.
3. **Model training & selection** — if applicable, trains the model with different parameters, evaluates performance, and selects the best one. Trains a final model and **saves it to disk** for later inference.

> ⚠️ Since we are not using an ML Engineering Framework, consider a good standard for saving trained models so we can maintain a **history of model versions**.