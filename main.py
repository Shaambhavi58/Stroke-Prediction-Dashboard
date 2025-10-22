
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix

def ensure_dirs():
    os.makedirs("output_graphs", exist_ok=True)

def load_data(path="stroke_data.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Make sure the CSV is in this folder.")
    df = pd.read_csv(path)
    return df

def preprocess(df):
    X = pd.get_dummies(
        df[['age','hypertension','heart_disease','avg_glucose_level','bmi','smoking_status','gender']],
        drop_first=True
    )
    y = df['stroke']
    return X, y

def train_and_eval(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=y
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.3f})')
    plt.plot([0,1],[0,1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Stroke Prediction – ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig('output_graphs/roc_curve.png', bbox_inches='tight')
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print(f"AUC: {roc_auc:.4f}")

    return roc_auc

def main():
    ensure_dirs()
    df = load_data()
    # quick EDA plot
    plt.figure()
    df['bmi'].hist(bins=30)
    plt.xlabel('BMI'); plt.ylabel('Count'); plt.title('BMI Distribution')
    plt.savefig('output_graphs/bmi_distribution.png', bbox_inches='tight')
    plt.close()

    X, y = preprocess(df)
    train_and_eval(X, y)

if __name__ == "__main__":
    main()
