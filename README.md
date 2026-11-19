# Practical Assignment 5: Ensemble Learning and Performance Evaluation

## Overview

This project implements and compares three ensemble-learning
methods---**Stacking**, **AdaBoost**, and **majority voting**---with
four individual machine-learning classifiers using the built-in **Iris
dataset** from scikit-learn.

The experiment uses a stratified 80:20 train-test split and evaluates
each model with accuracy, weighted precision, weighted recall, weighted
F1-score, classification reports, and confusion matrices.

## Objectives

1.  Implement stacking with a Logistic Regression meta-classifier.
2.  Implement AdaBoost using decision-tree stumps as weak learners.
3.  Implement a hard-voting ensemble that combines predictions by
    majority vote.
4.  Compare ensemble models against individual base classifiers using
    classification metrics.
5.  Interpret the results and identify appropriate use cases and
    possible improvements.

## Dataset

The project uses `sklearn.datasets.load_iris()`, so no external dataset
download is required.

  Property              Description
  --------------------- ---------------------------------
  Dataset               Iris
  Samples               150
  Input features        4 numeric measurements
  Classes               Setosa, Versicolor, Virginica
  Training samples      120 (80%)
  Testing samples       30 (20%)
  Split configuration   `stratify=y`, `random_state=42`

## Models Implemented

### Individual base classifiers

-   Logistic Regression
-   Decision Tree (maximum depth 3)
-   K-Nearest Neighbors (5 neighbors)
-   Support Vector Machine (RBF kernel)

Feature scaling is applied through `StandardScaler` pipelines for
Logistic Regression, KNN, and SVM. The Decision Tree does not require
feature scaling.

### Ensemble classifiers

-   **Stacking:** Uses the four base classifiers and a Logistic
    Regression meta-classifier. Five-fold cross-validation generates
    out-of-fold predictions; `predict_proba` is used as the stacking
    method.
-   **AdaBoost:** Uses 50 decision-tree stumps (`max_depth=1`) with a
    learning rate of 1.0.
-   **Hard Voting:** Combines the four base classifiers by selecting the
    class with the largest number of votes.

## Requirements

-   Python 3.9 or later recommended
-   scikit-learn
-   pandas
-   NumPy
-   Matplotlib

Install the dependencies:

``` bash
python -m pip install scikit-learn pandas numpy matplotlib
```

## Project Files

``` text
.
├── ensemble_learning_iris.py
├── README.md
├── Practical_Assignment_5_Ensemble_Learning_Iris.docx
├── ensemble_metrics_comparison.png
└── ensemble_confusion_matrices.png
```

The Python script is the executable implementation. The DOCX file is the
practical-assignment report. The two PNG files are the metric-comparison
and ensemble confusion-matrix figures.

## How to Run

1.  Open a terminal in the project directory.
2.  Install the required libraries.
3.  Run:

``` bash
python ensemble_learning_iris.py
```

The script loads the built-in Iris dataset, trains all seven models, and
prints each model's classification report and confusion matrix, followed
by a consolidated performance-comparison table.

You can also run the script in Jupyter Notebook, Google Colab, or VS
Code after installing the dependencies.

## Experimental Results

Results for the specified split and random seed:

  ------------------------------------------------------------------------
  Model              Accuracy       Weighted       Weighted       Weighted
                                   Precision         Recall       F1-score
  ------------ -------------- -------------- -------------- --------------
  Logistic             0.9333         0.9333         0.9333         0.9333
  Regression                                                

  Decision             0.9667         0.9697         0.9667         0.9666
  Tree                                                      

  K-Nearest            0.9333         0.9444         0.9333         0.9327
  Neighbors                                                 

  Support              0.9667         0.9697         0.9667         0.9666
  Vector                                                    
  Machine                                                   

  Stacking             0.9333         0.9333         0.9333         0.9333

  AdaBoost             0.9333         0.9333         0.9333         0.9333

  Hard Voting          0.9333         0.9333         0.9333         0.9333
  ------------------------------------------------------------------------

**Observation:** Decision Tree and SVM achieved the highest accuracy
(96.67%) on this particular test split. Each ensemble method achieved
93.33%. This demonstrates that an ensemble is not guaranteed to
outperform the strongest individual classifier on every dataset or
split.

## Evaluation Metrics

-   **Accuracy:** Fraction of all test predictions classified correctly.
-   **Precision (weighted):** Weighted average of class-wise precision.
-   **Recall (weighted):** Weighted average of class-wise recall.
-   **F1-score (weighted):** Weighted average of class-wise F1-scores.
-   **Confusion matrix:** Counts of actual classes against predicted
    classes.

Weighted metrics are used because they account for the number of test
examples in each class. For imbalanced datasets, macro-F1, per-class
recall, and balanced accuracy should also be considered.

## When to Use Each Ensemble Method

  -----------------------------------------------------------------------
  Method                  Suitable when           Main consideration
  ----------------------- ----------------------- -----------------------
  Stacking                Base models are diverse More computationally
                          and there is enough     involved; validate the
                          data to train a         meta-model carefully
                          reliable                
                          meta-classifier         

  AdaBoost                Weak learners can       Can be sensitive to
                          improve by focusing     noisy labels and
                          sequentially on earlier outliers
                          mistakes                

  Hard Voting             Several useful,         Equal votes ignore
                          complementary           differences in
                          classifiers are         classifier confidence
                          available and a simple  and reliability
                          combination is desired  
  -----------------------------------------------------------------------

## Limitations and Future Improvements

-   Iris is a small, balanced, relatively clean benchmark dataset;
    results may not transfer to larger real-world problems.
-   A single 30-sample test set gives a limited estimate of
    generalization performance.
-   Use repeated stratified cross-validation for more stable
    comparisons.
-   Tune hyperparameters using cross-validation confined to the training
    data.
-   Compare hard voting with soft voting and weighted voting.
-   Experiment with alternative base estimators and meta-classifiers.
-   Evaluate robustness to noisy data, missing values, and class
    imbalance.
-   Consider inference time, memory use, interpretability, and the cost
    of different classification errors.

## Reproducibility

The train-test split and stochastic estimators use `random_state=42`.
Results can vary if the split, model settings, or library versions are
changed. For a stronger performance claim, use repeated cross-validation
and retain a final untouched test set.

## Author / Student Details

Complete the student name, PRN, division, batch, roll number, faculty
name, and submission details in the practical report before submission.

## License

This project is prepared for academic practical-assignment use. Add a
license if you intend to distribute or reuse the code publicly.
