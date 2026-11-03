# Spam Message Classifier

A machine learning project developed as part of the **CodeNova Internship Program – Batch 01 (Task 2)**.

This project classifies SMS messages as either **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and machine learning techniques.

## Project Overview

The goal is to build and evaluate text classification models that can identify spam messages while minimizing false spam detections.

Two machine learning algorithms were trained and compared:

- Multinomial Naive Bayes
- Logistic Regression

TF-IDF (Term Frequency–Inverse Document Frequency) was used to convert text messages into numerical features.

## Dataset

- **Source:** [India Spam SMS Classification – Kaggle](https://www.kaggle.com/datasets/junioralive/india-spam-sms-classification)
- **Original records:** 2,267
- **Records after cleaning:** 2,062
- **Columns:** `Msg`, `Label`
- **Classes:** `ham`, `spam`

### Data Cleaning

- Removed 1 row containing a missing message.
- Removed 204 duplicate records.
- Reset the dataset index.
- Retained short messages and potentially useful numbers and special characters.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib
- Jupyter Notebook

## Methodology

1. Loaded and explored the SMS dataset.
2. Cleaned missing and duplicate records.
3. Examined the class distribution and message characteristics.
4. Split the data into training (80%) and testing (20%) sets using stratified sampling.
5. Applied TF-IDF vectorization with unigrams and bigrams.
6. Trained Multinomial Naive Bayes and Logistic Regression pipelines.
7. Evaluated both models using accuracy, precision, recall, F1-score, and confusion matrices.
8. Selected Logistic Regression as the final model and saved it using Joblib.

## Model Performance

Both models were evaluated on the same held-out test set of 413 messages.

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1-Score |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 98.31% | 97.95% | 97.28% | 97.61% |
| Logistic Regression | 98.79% | 100.00% | 96.60% | 98.27% |

### Final Model

**Logistic Regression** was selected because it achieved the highest accuracy, spam precision, and spam F1-score.

Its 100% spam precision on this test split means that none of the messages predicted as spam in this evaluation were actually ham. This result applies to the held-out test set and does not guarantee perfect performance on unseen real-world messages.

## Sample Predictions

| Message | Prediction |
|---|---|
| URGENT! You have won Rs. 50000. Click here to claim your reward! | Spam |
| Good morning! I will reach the office in 10 minutes. | Ham |
| Your OTP is 458921. Do not share it with anyone. | Ham |
| Congratulations! Claim your free gift now! | Spam |

## Project Structure

```text
task-2-spam-classifier/
│
├── data/
│   └── spam_ham_india.csv
│
├── figures/
│   ├── label_distribution.png
│   ├── naive_bayes_confusion_matrix.png
│   ├── logistic_regression_confusion_matrix.png
│   └── model_comparison.png
│
├── models/
│   └── spam_classifier_model.joblib
│
├── spam_classifier.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd task-2-spam-classifier
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

Open `spam_classifier.ipynb` in Jupyter Notebook or VS Code and run the cells.

## Using the Saved Model

The trained Logistic Regression pipeline can be loaded and used to classify new messages:

```python
import joblib

model = joblib.load("models/spam_classifier_model.joblib")

message = "Congratulations! You have won a free gift."
prediction = model.predict([message])[0]

print("Prediction:", prediction.upper())
```

## Author

Developed as part of the **CodeNova Internship Program – Batch 01**.

## Dataset Attribution

Dataset: [India Spam SMS Classification](https://www.kaggle.com/datasets/junioralive/india-spam-sms-classification)

Dataset licence: MIT License. Please refer to the original Kaggle dataset page for the licence and attribution details.