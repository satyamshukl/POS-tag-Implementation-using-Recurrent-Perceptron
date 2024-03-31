import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from model import *
from utility import *


def train(train_data, test_data, epochs, learning_rate=0.1, momentum=0.9, threshold=0.5):
    model = Model(learning_rate=learning_rate, momentum=momentum, threshold=threshold)

    # Initialize lists to store performance metrics for each epoch
    epoch_train_losses = []
    epoch_train_accuracies = []
    epoch_train_precisions = []
    epoch_train_recalls = []
    epoch_train_f1_scores = []

    epoch_test_losses = []
    epoch_test_accuracies = []
    epoch_test_precisions = []
    epoch_test_recalls = []
    epoch_test_f1_scores = []

    for epoch in range(epochs):
        total_train_loss = 0
        train_predictions, train_truth = [], []
        for x in train_data:
            y_preds, loss = model.forward(x, train=True)
            y = x['chunk_tags']
            train_predictions += y_preds
            train_truth += y
            total_train_loss += loss
            model.backward()

        total_train_loss /= len(train_data)
        train_precision, train_recall, train_accuracy, train_f1score = model.evaluate(train_predictions, train_truth)

        # Save performance metrics for the training data
        epoch_train_losses.append(total_train_loss)
        epoch_train_accuracies.append(train_accuracy)
        epoch_train_precisions.append(train_precision)
        epoch_train_recalls.append(train_recall)
        epoch_train_f1_scores.append(train_f1score)

        # Evaluate on test data after each epoch
        total_test_loss = 0
        test_predictions, test_truth = [], []
        wrong_predictions_test = []
        for x in test_data:
            y_preds, loss = model.forward(x)
            test_predictions += y_preds
            test_truth += x['chunk_tags']
            total_test_loss += loss

            if y_preds != x['chunk_tags']:  # If prediction is wrong
                wrong_predictions_test.append({
                    "tokens": x["tokens"],
                    "pos_tags": x["pos_tags"],
                    "actual_chunk_tags": x["chunk_tags"],
                    "predicted_chunk_tags": y_preds
                })

        total_test_loss /= len(test_data)
        test_precision, test_recall, test_accuracy, test_f1score = model.evaluate(test_predictions, test_truth)
        classification_report_str = classification_report(test_truth, test_predictions, output_dict=True)
        save_classification_report(classification_report_str, f'test_classification_report')
        save_wrong_predictions(wrong_predictions_test, f'wrong_predictions_test')

        # Save performance metrics for the test data
        epoch_test_losses.append(total_test_loss)
        epoch_test_accuracies.append(test_accuracy)
        epoch_test_precisions.append(test_precision)
        epoch_test_recalls.append(test_recall)
        epoch_test_f1_scores.append(test_f1score)

        print(f"Epoch {epoch + 1}/{epochs}:")
        print("Training Metrics:")
        model.print_score(train_precision, train_recall, train_accuracy, train_f1score, total_train_loss)
        print("Testing Metrics:")
        model.print_score(test_precision, test_recall, test_accuracy, test_f1score, total_test_loss)

    model.load_weights()

    # Plot performance metrics across all epochs for training data
    plot_performance_metrics(epoch_train_losses, "Loss", "train_loss")
    plot_performance_metrics(epoch_train_accuracies, "Accuracy", "train_accuracy")
    plot_performance_metrics(epoch_train_precisions, "Precision", "train_precision")
    plot_performance_metrics(epoch_train_recalls, "Recall", "train_recall")
    plot_performance_metrics(epoch_train_f1_scores, "F1-score", "train_f1_score")

    # Plot performance metrics across all epochs for test data
    plot_performance_metrics(epoch_test_losses, "Loss", "test_loss")
    plot_performance_metrics(epoch_test_accuracies, "Accuracy", "test_accuracy")
    plot_performance_metrics(epoch_test_precisions, "Precision", "test_precision")
    plot_performance_metrics(epoch_test_recalls, "Recall", "test_recall")
    plot_performance_metrics(epoch_test_f1_scores, "F1-score", "test_f1_score")


if __name__ == "__main__":
    with open('train.jsonl') as f:
        train_data = [json.loads(line) for line in f]

    with open('test.jsonl') as f:
        test_data = [json.loads(line) for line in f]

    train(train_data=train_data, test_data=test_data, epochs=10, learning_rate=0.01, momentum=0.009, threshold=0.5)
