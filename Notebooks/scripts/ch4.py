import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(confusion_matrix):
    """plot a confusion matrix with labels"""
    
    ax = sns.heatmap(pd.DataFrame(confusion_matrix),
                     annot=True, cmap="Blues", cbar=False, fmt='g')
    plt.xlabel("Predicted label", fontsize = 15)
    plt.ylabel("True label",    fontsize = 15)
    plt.show()

# Classifier stats
# -------------------------------------------------
def accuracy(confusion_matrix):
    """Overall, how often is the classifier correct?
        ex:
        x = confusion_matrix(y_true, y_pred)
        ... array([[0, 1],
                   [2, 3]], dtype=int64)
        accuracy(x)= (0+3)/(0+1+2+3)"""
    return np.trace(confusion_matrix) / np.sum(confusion_matrix)

def total_error_rate(confusion_matrix):
    """total error rate from confusion matrix
        ex:
        x = confusion_matrix(y_true, y_pred)
        ... array([[0, 1],
                   [2, 3]], dtype=int64)
        total_error_rate(x)= 1 - (0+3)/(0+1+2+3)"""
    return 1 - accuracy(confusion_matrix)


def true_negative_rate(confusion_matrix):
    """also known as "Specificity"
       When it's actually no, how often does it predict no?
        ex:
        x = confusion_matrix(y_true, y_pred)
        ... array([[0, 1],
                   [2, 3]], dtype=int64)
        true_negative_rate(x)= (0)/(0+1)"""
    return confusion_matrix[0, 0] / np.sum(confusion_matrix[0, :])

def true_positive_rate(confusion_matrix):
    """also known as "Sensitivity" or "Recall"
       When it's actually yes, how often does it predict yes?
        ex:
        x = confusion_matrix(y_true, y_pred)
        ... array([[0, 1],
                   [2, 3]], dtype=int64)
        true_positive_rate(x)= (3)/(2+3)"""
    return confusion_matrix[1, 1] / np.sum(confusion_matrix[1, :])

def false_negative_rate(confusion_matrix):
    """When it's actually yes, how often does it predict no?
        ex:
        x = confusion_matrix(y_true, y_pred)
        ... array([[0, 1],
                   [2, 3]], dtype=int64)
        false_negative_rate(x)= (2)/(2+3)"""
    return 1 - true_positive_rate(confusion_matrix)

def false_positive_rate(confusion_matrix):
    """When it's actually no, how often does it predict yes?
        ex:
        x = confusion_matrix(y_true, y_pred)
        ... array([[0, 1],
                   [2, 3]], dtype=int64)
        false_positive_rate(x)= (1)/(0+1)"""
    return 1 - true_negative_rate(confusion_matrix)

def positive_predictive_value(confusion_matrix):
    """also known as "Precision": When it predicts yes, how often is it correct?
        ex:
        x = confusion_matrix(y_true, y_pred)
        ... array([[0, 1],
                   [2, 3]], dtype=int64)
        positive_predictive_value(x)= (3)/(1+3)"""
    return confusion_matrix[1, 1] / np.sum(confusion_matrix[:, 1])

def negative_predictive_value(confusion_matrix):
    """the proportion of predicted negatives that are correctly predicted
        ex:
        x = confusion_matrix(y_true, y_pred)
        ... array([[0, 1],
                   [2, 3]], dtype=int64)
        negative_predictive_value(x)= (0)/(0+2)"""
    return confusion_matrix[0, 0] / np.sum(confusion_matrix[:, 0])

def prior_error_rate(confusion_matrix):
    """The prior probability that a result is positive
        ex:
        x = confusion_matrix(y_true, y_pred)
        ... array([[0, 1],
                   [2, 3]], dtype=int64)
        prior_error_rate(x)= 1 - (2+3)/(0+1+2+3)"""
    return 1 - (np.sum(confusion_matrix[1, :]) / np.sum(confusion_matrix))

def conf_mtx_stats(confusion_matrix):
    return round(pd.Series({'Accuracy': accuracy(confusion_matrix),
                      'Total_Error_Rate': total_error_rate(confusion_matrix),
                      'True_Negative_Rate (Specificity)': true_negative_rate(confusion_matrix),
                      'True_Positive_Rate (Recall)': true_positive_rate(confusion_matrix),
                      'False_Negative_Rate': false_negative_rate(confusion_matrix),
                      'False_Positive_Rate': false_positive_rate(confusion_matrix),
                      'negative_Predictive_Value': negative_predictive_value(confusion_matrix),
                      'Positive_Predictive_Value (Precision)': positive_predictive_value(confusion_matrix),
                      'Prior_Error_Rate': prior_error_rate(confusion_matrix),
                      }),4)