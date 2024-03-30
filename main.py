import json
from sklearn.model_selection import train_test_split
from model import *

def train(train_data, test_data, epochs, learnin_rate=1.0, threshold=0.0):
    model = Model(learning_rate=learnin_rate, threshold=threshold) 

    for epoch in range(epochs):
        total_loss = 0
        predictions, truth = [], []
        for x in train_data:
            y_preds, loss = model.forward(x, train=True)
            y = x['chunk_tags']
            predictions += y_preds
            truth += y
            total_loss += loss
            model.backward()
        
        total_loss /= len(train_data)
        precision, recall, accuracy, f1score = model.evaluate(predictions, truth)
        print(f"\n\nScore at epoch : {epoch}")
        model.print_score(precision, recall, accuracy, f1score, total_loss)

    model.load_weights()

    model.test(test_data)
    print('Model Weights:')
    print(model.weights)


if __name__ == "__main__":
    with open('train.jsonl') as f:
        train_data = [json.loads(line) for line in f]

    with open('test.jsonl') as f:
        test_data = [json.loads(line) for line in f]

    train(train_data=train_data, test_data=test_data, epochs=3, learnin_rate=0.1, threshold=0.5)

    # model.train(train_data, epochs)
    # model.load_weights()
    
    # model.test(test_data)
    # print('Model Weights:')
    # print(model.weights)