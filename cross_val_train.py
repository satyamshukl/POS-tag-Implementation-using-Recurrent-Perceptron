import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from model import *
from utility import *


def train(train_data, test_data, epochs, learning_rate=1.0, threshold=0.0):
    kf = KFold(n_splits=5, shuffle=True)
    
    # Initialize lists to store performance metrics for each fold
    fold_losses = []
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1_scores = []

    weights = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_data), 1):
        print('_'*80)
        print(f"\nTraining Fold {fold}\n")

        train_fold = [train_data[i] for i in train_idx]
        val_fold = [train_data[i] for i in val_idx]

        model = Model(learning_rate=learning_rate, momentum=0.9, threshold=threshold) 

        epoch_losses = []
        epoch_accuracies = []
        epoch_precisions = []
        epoch_recalls = []
        epoch_f1_scores = []

        wrong_predictions_train = []  # To store instances where the model prediction was wrong for train data
        wrong_predictions_val = []    # To store instances where the model prediction was wrong for validation data

        for epoch in range(epochs):
            total_loss = 0
            predictions, truth = [], []
            for x in train_fold:
                y_preds, loss = model.forward(x, train=True)
                y = x['chunk_tags']
                predictions += y_preds
                truth += y
                total_loss += loss
                model.backward()

            total_loss /= len(train_fold)
            precision, recall, accuracy, f1score = model.evaluate(predictions, truth)
            print(f"Epoch {epoch + 1}/{epochs}:")
            model.print_score(precision, recall, accuracy, f1score, total_loss)

            # Append performance metrics for the current epoch
            epoch_losses.append(total_loss)
            epoch_accuracies.append(accuracy)
            epoch_precisions.append(precision)
            epoch_recalls.append(recall)
            epoch_f1_scores.append(f1score)

            # Save classification report after the final epoch for validation data
            if epoch == epochs - 1:
                val_predictions = []
                val_truth = []
                for x in val_fold:
                    y_preds, _ = model.forward(x)
                    val_predictions += y_preds
                    val_truth += x['chunk_tags']
                    if y_preds != x['chunk_tags']:  # If prediction is wrong
                        wrong_predictions_val.append({
                            "tokens": x["tokens"],
                            "pos_tags": x["pos_tags"],
                            "actual_chunk_tags": x["chunk_tags"],
                            "predicted_chunk_tags": y_preds
                        })

                classification_report_str = classification_report(val_truth, val_predictions, output_dict=True)
                print(classification_report(val_truth, val_predictions))
                save_classification_report(classification_report_str, f'classification_report_fold_{fold}')

        # Append performance metrics for the current fold
        fold_losses.append(epoch_losses)
        fold_accuracies.append(epoch_accuracies)
        fold_precisions.append(epoch_precisions)
        fold_recalls.append(epoch_recalls)
        fold_f1_scores.append(epoch_f1_scores)

        weights.append(model.weights)
        
        os.system('python3 condition.py')

        model.save_weights(cross_val=True)

        print(f"\nValidation Score for Fold {fold}:")
        model.test(val_fold)

        print('\nModel Weights:')
        print(model.weights.reshape(1, 11))
        # os.system("python3 condition.py")
        
        # Save wrong predictions to a JSON file for validation data
        save_wrong_predictions(wrong_predictions_val, f'wrong_predictions_fold_{fold}')

    # Plot performance metrics for each fold and save the plots
    plot_performance_metrics(fold_losses, "Loss", f"cross_val_train_loss_fold_{fold}", range(1, epochs+1))
    plot_performance_metrics(fold_accuracies, "Accuracy", f"cross_val_train_accuracy_fold_{fold}", range(1, epochs+1))
    plot_performance_metrics(fold_precisions, "Precision", f"cross_val_train_precision_fold_{fold}", range(1, epochs+1))
    plot_performance_metrics(fold_recalls, "Recall", f"cross_val_train_recall_fold_{fold}", range(1, epochs+1))
    plot_performance_metrics(fold_f1_scores, "F1-score", f"cross_val_train_f1_score_fold_{fold}", range(1, epochs+1))

    # Calculate average performance metrics across all folds
    avg_loss = np.mean(fold_losses, axis=0)
    avg_accuracy = np.mean(fold_accuracies, axis=0)
    avg_precision = np.mean(fold_precisions, axis=0)
    avg_recall = np.mean(fold_recalls, axis=0)
    avg_f1_score = np.mean(fold_f1_scores, axis=0)

    # Print average performance metrics
    print("\nAverage Performance Metrics Across All Folds:")
    print(f"Average Loss: {avg_loss[-1]}")
    print(f"Average Accuracy: {avg_accuracy[-1]}")
    print(f"Average Precision: {avg_precision[-1]}")
    print(f"Average Recall: {avg_recall[-1]}")
    print(f"Average F1-score: {avg_f1_score[-1]}")

    # Update model weights as the average of all fold weights
    model.weights = np.mean([weight for weight in weights], axis=0)
    model.save_weights(cross_val=True)

    print()
    print("*"*80)
    print("Testing on Test Data...")
    model.test(test_data)
    print("*"*80)

    # Save classification report for test data
    test_predictions = []
    test_truth = []
    wrong_predictions_test = []


    for x in test_data:
        y_preds, _ = model.forward(x)
        test_predictions += y_preds
        test_truth += x['chunk_tags']
        if y_preds != x['chunk_tags']:  # If prediction is wrong
            wrong_predictions_test.append({
                "tokens": x["tokens"],
                "pos_tags": x["pos_tags"],
                "actual_chunk_tags": x["chunk_tags"],
                "predicted_chunk_tags": y_preds
            })

    classification_report_str = classification_report(test_truth, test_predictions, output_dict=True)
    
    print(classification_report(test_truth, test_predictions))
    save_classification_report(classification_report_str, 'classification_report_cross_val_test')
    save_wrong_predictions(wrong_predictions_test, 'wrong_predictions_cross_val_test')


if __name__ == "__main__":
    with open('filtered_output.jsonl') as f:
        train_data = [json.loads(line) for line in f]

    with open('test.jsonl') as f:
        test_data = [json.loads(line) for line in f]

    train(train_data=train_data, test_data=test_data, epochs=3, learning_rate=0.1, threshold=0.5)
