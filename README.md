# MediPredict AI – AI-Based Disease Prediction and Clinical Decision Support System

MediPredict AI is a machine learning-based web application that predicts possible diseases based on user-selected symptoms. It combines a traditional machine learning pipeline with a Flask-powered web interface to provide preliminary health-related insights and educational disease information.

The project demonstrates how machine learning can be used to analyze symptom patterns and support preliminary exploration of possible health conditions.

> **Disclaimer:** MediPredict AI is an educational prototype, not a medical diagnostic tool. Its predictions must not be used as a substitute for professional medical advice, diagnosis, or treatment.

---

## Features

- **Symptom-Based Prediction:** Select one or more symptoms from an available list of 131 symptoms.
- **Machine Learning Models:** Implements and compares four traditional machine learning algorithms.
- **Model Evaluation:** Evaluates models using accuracy, precision, recall, F1-score, and cross-validation.
- **Disease Information:** Displays dataset-provided descriptions and precaution information for predicted diseases.
- **Interactive Web Interface:** Flask-based application with a responsive HTML, CSS, and JavaScript frontend.
- **Symptom Search:** Search and select symptoms through an interactive interface.
- **Prediction Summary:** Displays the predicted disease and the symptoms selected by the user.

---

## Machine Learning Approach

The system uses a supervised multi-class classification approach.

### Dataset

The project uses the [Disease Symptom Prediction dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) from Kaggle.

The dataset contains:

| Attribute | Value |
|---|---:|
| Original records | 4,920 |
| Disease classes | 41 |
| Symptom columns | 17 |
| Unique symptoms after preprocessing | 131 |
| Unique symptom combinations after deduplication | 304 |

### Data Preprocessing

The preprocessing pipeline includes:

1. Loading and inspecting the dataset.
2. Handling missing symptom entries.
3. Normalizing symptom names.
4. Identifying and removing duplicate symptom combinations.
5. Converting symptoms into binary features using `MultiLabelBinarizer`.
6. Splitting the processed data into training and testing sets using stratified sampling.

The final dataset contains 304 unique symptom combinations represented by 131 binary features.

### Machine Learning Algorithms

Four algorithms were implemented and compared:

- Decision Tree Classifier
- Random Forest Classifier
- Multinomial Naive Bayes
- Support Vector Machine (SVM)

The models were evaluated using a stratified train-test split and 5-fold stratified cross-validation.

**Random Forest Classifier** was selected as the final prediction model based on the evaluation results.

### Model Evaluation

The following metrics were used:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix
- 5-Fold Cross-Validation

The evaluation results are available in the `reports/` directory.

**Evaluation note:** The dataset is highly repetitive and contains only 304 unique symptom combinations after deduplication. The strong evaluation scores reflect performance on this curated dataset and do not establish real-world clinical accuracy or generalization.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Model Visualization | Matplotlib, Seaborn |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Model Serialization | Joblib |
| Development Environment | VS Code |

---

## Project Structure

```text
MediPredict-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── dataset.csv
│   ├── Symptom-severity.csv
│   ├── symptom_Description.csv
│   └── symptom_precaution.csv
│
├── models/
│   ├── best_model.pkl
│   └── symptom_binarizer.pkl
│
├── src/
│   ├── data_inspection.py
│   ├── duplicate_analysis.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── validate_models.py
│   ├── pattern_analysis.py
│   ├── feature_analysis.py
│   ├── predict.py
│   └── disease_information.py
│
├── templates/
│   ├── index.html
│   ├── predictor.html
│   └── about.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── predictor.js
│   └── images/
│
├── reports/
│   ├── classification_report.txt
│   ├── confusion_matrix.csv
│   ├── cross_validation_results.csv
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   └── model_comparison.csv
│
└── notebooks/
```

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Ayushi081005/MediPredict-AI.git
```

### 2. Navigate to the project directory

```bash
cd MediPredict-AI
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Flask application

```bash
python app.py
```

### 7. Open the application

Visit:

```text
http://127.0.0.1:5000/
```

---

## How It Works

1. The user opens the MediPredict AI web application.
2. The user navigates to the symptom prediction page.
3. The user searches for and selects relevant symptoms.
4. The selected symptoms are normalized and converted into binary feature vectors.
5. The trained Random Forest model predicts a disease class.
6. The application displays the predicted class along with available disease information from the supporting dataset.

---

## Limitations

- The system uses a limited, curated symptom dataset.
- Duplicate symptom combinations were removed to reduce repetition and avoid treating repeated records as independent examples.
- The processed dataset contains only 304 unique symptom combinations across 41 disease classes.
- Predictions depend entirely on the symptoms represented in the training dataset.
- The system does not consider patient age, medical history, physical examination, laboratory tests, or other clinical factors.
- The model has not been clinically validated on independent patient data.
- Dataset-provided precautions are informational and are not personalized treatment recommendations.

---

## Future Enhancements

- Expand the dataset with diverse and independently validated medical data.
- Improve model generalization through external validation.
- Add more robust handling of ambiguous or incomplete symptom inputs.
- Introduce explainability techniques for understanding model predictions.
- Deploy the application on a cloud hosting platform.

---

## Author

**Ayushi Shrivastava**

B.Tech – Computer Science and Engineering (AIML)

GitHub: [Ayushi081005](https://github.com/Ayushi081005)

---

## Acknowledgements

- Kaggle – Disease Symptom Prediction dataset.
- Scikit-learn documentation.
- Flask documentation.

---

**MediPredict AI – Exploring Machine Learning for Preliminary Symptom-Based Disease Prediction.**