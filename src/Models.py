import pandas, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

# --- Load all four datasets ---
df_2f_box       = pandas.read_csv(os.path.join(DATA_DIR, "grasp_results_two_finger_box.csv"))
df_2f_cylinder  = pandas.read_csv(os.path.join(DATA_DIR, "grasp_results_two_finger_cylinder.csv"))
df_3f_box       = pandas.read_csv(os.path.join(DATA_DIR, "grasp_results_three_finger_box.csv"))
df_3f_cylinder  = pandas.read_csv(os.path.join(DATA_DIR, "grasp_results_three_finger_cylinder.csv"))

# Put datasets into dictionary for looping
datasets = {
    "Two Finger – Box": df_2f_box,
    "Two Finger – Cylinder": df_2f_cylinder,
    "Three Finger – Box": df_3f_box,
    "Three Finger – Cylinder": df_3f_cylinder
}

# --- Label encoding shared across all datasets ---
all_results = pandas.concat([
    df_2f_box["Result"],
    df_2f_cylinder["Result"],
    df_3f_box["Result"],
    df_3f_cylinder["Result"]
])

encoder = LabelEncoder()
encoder.fit(all_results)

# --- Train a logistic regression model for each dataset ---
for name, df in datasets.items():

    # Add encoded label column
    df["label"] = encoder.transform(df["Result"])

    # Features: use XYZ + quaternion (X,Y,Z,W)
    X = df[["PosX","PosY","PosZ","AngX","AngY","AngZ","AngW"]]
    y = df["label"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=126, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Logistic regression
    model = LogisticRegression(C=0.01, max_iter=1000)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Print accuracy
    print(f"Accuracy ({name}): {acc}")

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix – {name}")
    plt.show()
