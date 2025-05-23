# 🚀 Addiction Risk Prediction

This repository provides a comprehensive machine learning pipeline designed to **predict addiction risk** using behavioral, psychological, and demographic data. The solution empowers early identification of high-risk individuals and supports data-driven intervention strategies.

---

## 🎯 Project Objectives

The goal of this project is to **predict the likelihood of addiction (Yes/No)** based on user-specific inputs including:

- Substance usage (Alcohol, Cannabis, Tobacco, etc.)
- Age of first use and current age
- Frequency of usage
- Reported stress levels and diagnosed mental health conditions
- Coping mechanisms and presence of support systems

Through this, we aim to:

- Understand patterns of substance exposure, especially in adolescence or pre-teen years
- Explore behavioral risk factors contributing to addiction
- Enable predictive modeling to assist mental health professionals

---

## 📊 Dataset Overview

The dataset was **collected firsthand using Google Forms**, ensuring authentic, real-world insights into individual behavioral patterns.

- 📎 [Google Form Link (Original Survey)](https://docs.google.com/forms/d/1YirjvdKbzAlse9nt-s2LNh7qeZnqBJoD0NLlJWJeRW8/edit#response=ACYDBNii8NpgF1WNf6UJR4RTrWEx3TEGfJBicwoyPBN89azQt3NawTW-Oa6A7EIp_-P49ms)

### 📁 Files Provided
- `Dataset.xlsx` – Raw collected data
- `Cleaned_Encoded_Dataset.xlsx` – Preprocessed dataset used for modeling
- `New.xlsx` – Data used for inference or testing predictions

---

## ⚙️ How to Run the Project

### ✅ Step-by-Step Instructions

1. **Launch the Notebook**
   - Open `Addiction_Risk_Prediction.ipynb`
   - Click “Open in Colab” for browser-based execution

2. **Upload the Required Files**
   - Download and upload the following to Colab:
     - `Dataset.xlsx`
     - `Cleaned_Encoded_Dataset.xlsx`
     - `New.xlsx`

3. **Run the Notebook**
   - Execute each cell sequentially to perform:
     - Data preprocessing
     - Feature engineering
     - Model training and evaluation
     - Final binary prediction (`Addiction: Yes / No`)

---

## 🤖 Machine Learning Approach

The project initially uses **multiclass classification** on the `substances_used` variable, which categorizes different types of substances. Post-training, the focus shifts to a **binary classification task** to determine **whether an individual is at risk of addiction**.

### 🔍 Models Benchmarked

- Random Forest ✅ *(Best Performing Model)*
- Support Vector Machine (SVM)
- XGBoost
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Naive Bayes

### 🧠 Ensemble Model
We also developed a **stacked ensemble model** to boost performance:
- **Base Models:** Random Forest, SVM, XGBoost
- **Meta-Learner:** Logistic Regression

Performance was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score

---

## 📌 Real-World Insight

Our findings suggest that:
- Individuals are often exposed to addictive substances during **pre-teen and adolescent** years
- Stress levels, inadequate coping mechanisms, and mental health diagnoses are significant predictors
- Early intervention can be guided using such predictive models

---

## 📈 Future Roadmap

- Deploy as a REST API or interactive dashboard
- Integrate interpretability tools (e.g., SHAP, LIME)
- Automate data ingestion pipelines for real-time risk prediction
- Extend dataset for temporal analysis or time series modeling

---

## 📬 Contact

For contributions, queries, or to collaborate on health-tech initiatives, reach out via [GitHub Issues](https://github.com/your-username/Addiction-Risk-Prediction/issues).

---

## 📎 License

This project is licensed under the **MIT License**. Please review the LICENSE file for details.

---

_Developed with an intent to support mental health awareness, data literacy, and predictive analytics in public health._
