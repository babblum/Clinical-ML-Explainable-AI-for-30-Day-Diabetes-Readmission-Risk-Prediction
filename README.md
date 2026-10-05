# 🏥 Clinical ML & Explainable AI for 30-Day Diabetes Readmission Risk Prediction

A machine learning pipeline designed to predict 30-day hospital readmissions for patients with diabetes using clinical and administrative data. The project covers data cleaning, deduplication, clinical cohort filtering, feature engineering, and class imbalance handling.

---

## 📌 Project Overview

Hospital readmission within 30 days is a key metric for healthcare quality and cost. This project builds a binary classification pipeline to identify high-risk diabetes patients at discharge.

### Key Objectives

* **Data Preprocessing & Cleaning:** Handle missing values represented by `?`, drop high-missingness attributes, and resolve patient deduplication.


* **Cohort Definition:** Filter out non-viable discharge dispositions (e.g., hospice or expired patients).


* **Machine Learning Pipeline:** Prepare features for training using scikit-learn pipelines, cross-validation, and tree-based/linear models.


* **Model Evaluation & Interpretability:** Evaluate models using metrics suited for imbalanced healthcare datasets ($ROC\text{-}AUC$, $PR\text{-}AUC$, Calibration) and explain predictions using SHAP (SHapley Additive exPlanations).



---

## 📊 Dataset & Schema

The pipeline utilizes the **Diabetes 130-US hospitals for years 1999–2008** dataset.

* **Dataset Dimensions:** 101,766 rows × 50 columns


* **Target Variable (`readmitted`):** Converted to binary classification:


* `1`: Readmitted within 30 days (`<30`)


* `0`: Not readmitted within 30 days (`>30` or `NO`)




* **Class Imbalance:** ~8.93% positive class (30-day readmissions) vs. ~91.07% negative class.



### Primary Input Files

1. `diabetic_data.csv`: Primary clinical dataset.


2. `IDS_mapping.csv`: ID mappings for `admission_type_id`, `discharge_disposition_id`, and `admission_source_id`.



---

## 🛠 Data Pipeline & Methodology

### 1. Data Cleaning & Audit

* **Missing Value Standardisation:** Replaced `?` placeholders with `NaN`.


* **Deduplication:** Retained only the first chronological admission for patients with multiple visits (`patient_nbr`), dropping 30,248 duplicate records.


* **Cohort Exclusion:** Removed 1,084 records where discharge disposition indicated patient death or hospice care (IDs: `11, 19, 20, 21`).


* **Feature Dropping:** Excluded columns with >40% missing values (`weight`, `payer_code`, `medical_specialty`, `max_glu_serum`).



| Stage | Rows Remaining | Rows Removed | Columns |
| --- | --- | --- | --- |
| **Raw Dataset** | 101,766 | 0 | 50

 |
| **Deduplication (First Encounter)** | 71,518 | 30,248 | 50

 |
| **Hospice / Expired Removal** | 70,434 | 1,084 | 50

 |
| **High-Missingness Drop** | 70,434 | 0 | 46

 |

---

## ⚙️ Requirements & Installation

### Prerequisites

* Python 3.9+
* Anaconda or virtual environment (recommended)

### Installation

Clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/your-username/diabetes-readmission-prediction.git
cd diabetes-readmission-prediction

# Install requirements
pip install pandas numpy scikit-learn matplotlib seaborn xgboost shap

```

---

## 🚀 Usage

1. Place `diabetic_data.csv` and `IDS_mapping.csv` in the root project directory.


2. Launch Jupyter Notebook or JupyterLab:
```bash
jupyter notebook

```


3. Open and run the main notebook to execute the pipeline from data loading through model evaluation.



---

## 📦 Project Structure

```text
.
├── diabetic_data.csv          # Primary hospital encounter dataset
├── IDS_mapping.csv            # ID lookup tables for categorical IDs
├── notebook.ipynb             # Main Jupyter Notebook containing complete workflow
└── README.md                  # Project documentation

```

---

## 💡 Key Libraries Used

* **Data Processing:** `pandas`, `numpy`

* **Machine Learning:** `scikit-learn`, `xgboost`

* **Model Explainability:** `shap`

* **Visualization:** `matplotlib`, `seaborn`
