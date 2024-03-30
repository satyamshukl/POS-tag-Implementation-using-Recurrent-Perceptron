import json
from sklearn.model_selection import KFold
from model import *


def train(train_data, test_data, epochs, learning_rate=1.0, threshold=0.0):
    kf = KFold(n_splits=5, shuffle=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_data), 1):
        print(f"Training Fold {fold}")

        train_fold = [train_data[i] for i in train_idx]
        val_fold = [train_data[i] for i in val_idx]

        model = Model(learning_rate=learning_rate, threshold=threshold)

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
            print(f"\n\nScore at epoch {epoch} for Fold {fold}:")
            model.print_score(precision, recall, accuracy, f1score, total_loss)

        model.load_weights()

        print(f"\nValidation Score for Fold {fold}:")
        model.test(val_fold)

    print("\nTesting on Test Data:")
    model.test(test_data)
    print('Model Weights:')
    print(model.weights)


if __name__ == "__main__":
    with open('train.jsonl') as f:
        train_data = [json.loads(line) for line in f]

    with open('test.jsonl') as f:
        test_data = [json.loads(line) for line in f]

    train(train_data=train_data, test_data=test_data, epochs=3, learning_rate=0.1, threshold=0.5)
