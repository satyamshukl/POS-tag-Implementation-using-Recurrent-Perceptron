import os
import json
import pandas as pd
import matplotlib.pyplot as plt


def save_classification_report(classification_report_dict, *args):
    if len(args) == 1:
        file_name = args[0]
    elif len(args) == 2:
        file_name, fold = args
        file_name = f"{file_name}_fold_{fold}"
    else:
        raise ValueError("Invalid number of arguments provided to save_classification_report")
    
    os.makedirs('classification_report', exist_ok=True)

    df = pd.DataFrame(classification_report_dict).transpose()
    df.to_csv(f"classification_report/{file_name}_classification_report.csv", index=True)


def save_wrong_predictions(wrong_predictions, *args):
    if len(args) == 1:
        file_name = args[0]
    elif len(args) == 2:
        file_name, fold = args
        file_name = f"{file_name}_fold_{fold}"
    else:
        raise ValueError("Invalid number of arguments provided to save_wrong_predictions")
    
    os.makedirs('error_analysis', exist_ok=True)

    with open(f"error_analysis/{file_name}.json", "w") as f:
        json.dump(wrong_predictions, f, indent=4)


def plot_performance_metrics(metrics, metric_name, *args):
    plt.figure(figsize=(10, 6))
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    os.makedirs('plots', exist_ok=True)

    # print('args: ', args)
    # print('length of args: ', len(args))


    if len(args) > 1:
        file_name = args[0]
        for i, metric in enumerate(metrics, 1):
            plt.plot(metric, label=f"Fold {i}")
        plt.legend()
        plt.title(f"{metric_name} Across {file_name}")
        plt.grid(True)
        plt.savefig(f"plots/{file_name}_{metric_name.lower().replace(' ', '_')}.png")
    else:
        plt.title(f"{metric_name} Across Epochs")
        plt.plot(metrics)
        # plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig(f"plots/{str(args[0]).lower().replace(' ', '_')}.png")
    plt.close()
