# Practical Assignment 5: Ensemble Learning and Performance Evaluation
# Dataset: Iris (built-in scikit-learn dataset)

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# 1. Load the built-in Iris dataset (150 samples, 4 features, 3 classes).
iris = load_iris()
X, y = iris.data, iris.target

# 2. Create a reproducible, stratified 80:20 train-test split.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Define diverse base classifiers. Scaling is placed in pipelines for
#    distance/margin-based models; the decision tree does not require scaling.
base_estimators = [
    ("lr", make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000, random_state=42))),
    ("dt", DecisionTreeClassifier(max_depth=3, random_state=42)),
    ("knn", make_pipeline(StandardScaler(),
                          KNeighborsClassifier(n_neighbors=5))),
    ("svm", make_pipeline(StandardScaler(),
                          SVC(kernel="rbf", C=1.0, probability=True,
                              random_state=42)))
]

# 4. Build the ensemble classifiers.
stacking = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5,
    stack_method="predict_proba"
)

adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

voting = VotingClassifier(
    estimators=base_estimators,
    voting="hard"  # majority vote; each classifier contributes one vote
)

# 5. Evaluate base and ensemble models on the same held-out test set.
models = [
    ("Logistic Regression", base_estimators[0][1]),
    ("Decision Tree", base_estimators[1][1]),
    ("K-Nearest Neighbors", base_estimators[2][1]),
    ("Support Vector Machine", base_estimators[3][1]),
    ("Stacking", stacking),
    ("AdaBoost", adaboost),
    ("Hard Voting", voting)
]

rows = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (weighted)": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "Recall (weighted)": recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "F1-score (weighted)": f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
    })

    print(f"\n{name}")
    print(classification_report(
        y_test, y_pred, target_names=iris.target_names, zero_division=0
    ))
    print("Confusion matrix:\\n", confusion_matrix(y_test, y_pred))

results = pd.DataFrame(rows)
print("\nPERFORMANCE COMPARISON")
print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
