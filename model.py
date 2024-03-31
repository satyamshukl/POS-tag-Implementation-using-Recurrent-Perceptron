import numpy as np

np.random.seed(41)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import nltk
from nltk.tokenize import word_tokenize
# nltk.download('punkt') 
# nltk.download('averaged_perceptron_tagger')

class Model():
    def __init__(self, learning_rate=0.1, momentum=0.9, threshold=0.5) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weights = np.random.rand(11, 1)
        self.grad = np.zeros((11, 1))
        self.prev_grad = np.zeros((11, 1))
        self.threshold = threshold
        

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    
    def CrossEntropy(self, y, y_hat):
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        return (-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat))
    
    
    def CrossEntropy_grad(self, y, y_hat):
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        return -(y / y_hat) + ((1 - y) / (1 - y_hat))
    

    def sigmoid_grad(self, y_hat):
        return y_hat * (1  - y_hat)


    def evaluate(self, y_pred, y):
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        accuracy = accuracy_score(y, y_pred)
        f1score = f1_score(y, y_pred)

        return precision, recall, accuracy, f1score
    
    
    def preprocess_text(self, text):
        # Remove punctuation and convert to lowercase
        text = text.lower()
        tokens = word_tokenize(text)
        # Remove stopwords (common words like 'the', 'is', etc.)
        filtered_tokens = [word for word in tokens]

        return filtered_tokens
    

    def save_weights(self):
        np.save("Weights.npy", self.weights)


    def load_weights(self):
        self.weights = np.load(f"Weights.npy")


    def get_pos_tags(self, sentence):
        preprocessed_tokens = self.preprocess_text(sentence)
        pos_tags = nltk.pos_tag(preprocessed_tokens)
        convert = {"NN":1, "DT":2, "JJ":3}
        output = [convert.get(i[1], 4) for i in pos_tags]

        return output
    
    
    def print_score(self, precision, recall, accuracy, f1score, loss):
        print(f"Precision {precision:.4f}, Recall {recall:.4f}, Accuracy {accuracy:.4f}, F1-score {f1score:.4f}, Loss {loss:.4f}")


    def create_one_hot(self, n, max=4):
        x = np.zeros(max)
        x[n - 1] = 1

        return x
    

    def inference(self, sentence):
        pos_tags = self.get_pos_tags(sentence)
        y_preds = []
        sos = np.array([1])
        y_pred = 0
        for i in range(len(pos_tags)):
            if i == 0:
                one_hot_prev_pos_tag = np.zeros(4)
            else:
                sos = np.array([0])
                one_hot_prev_pos_tag = self.create_one_hot(pos_tags[i-1])

            one_hot_curr_pos_tag = self.create_one_hot(pos_tags[i])
            input = np.concatenate((np.array([y_pred]), sos, one_hot_prev_pos_tag, one_hot_curr_pos_tag, np.array([-1.0]))) 
            y_logit = np.matmul(input, self.weights)[0]
            y_pred = 1 if y_logit >= 0.5 else 0
            y_preds.append(y_pred)
        
        output = []
        s = ""
        for i, word in enumerate(sentence.split()):
            if y_preds[i]==1:
                if s:output.append(s)
                s = word
            else:
                s += "_" + word 
        if s:output.append(s)
        output = " ".join(output)

        return y_preds, output
    

    def forward(self, x, train=False):
        pos_tags, chunk_tags = x["pos_tags"], x["chunk_tags"]
        y, y_pred, sos, loss, accuracy = 0, 0, np.array([1]), 0, 0
        y_preds = []
        grad_y = np.zeros((11, 1))
        grad = np.zeros((11, 1))
        for i in range(len(pos_tags)):
            if i==0:
                one_hot_prev_pos_tag = np.zeros(4)
            else:
                sos = np.array([0])
                one_hot_prev_pos_tag = self.create_one_hot(pos_tags[i-1])

            one_hot_curr_pos_tag = self.create_one_hot(pos_tags[i])

            if train:
                input = np.concatenate((np.array([y]), sos, one_hot_prev_pos_tag, one_hot_curr_pos_tag, np.array([-1.0])))
            else:
                input = np.concatenate((np.array([y_pred]), sos, one_hot_prev_pos_tag, one_hot_curr_pos_tag, np.array([-1.0]))) 

            y = chunk_tags[i]
            z = np.matmul(input, self.weights)[0]
            z = np.clip(z, -10, 10)
            y_logit = self.sigmoid(z)
            y_pred = 1 if y_logit >= self.threshold else 0
            y_preds.append(y_pred)
            accuracy += 1 if y_pred==y else 0
            grad_y = np.clip(grad_y, -1, 1)
            grad_y = self.momentum * grad_y + input.reshape(11, 1)
            grad += self.CrossEntropy_grad(y, y_logit) * self.sigmoid_grad(y_logit) * grad_y
            loss += self.CrossEntropy(y, y_logit)
        
        grad /= len(pos_tags)
        self.grad = grad
        accuracy /= len(pos_tags)
        self.save_weights()

        return y_preds, loss / len(pos_tags)
    
    
    def backward(self):
        self.weights -= self.learning_rate * self.grad


    def train(self, data, epochs):
        for epoch in range(epochs):
            total_loss = 0
            predictions, truth = [], []
            for x in data:
                y_preds, loss = self.forward(x, train=True)
                y = x['chunk_tags']
                predictions += y_preds
                truth += y
                total_loss += loss
                self.backward()
            
            total_loss /= len(data)
            precision, recall, accuracy, f1score = self.evaluate(predictions, truth)
            print(f"\n\nScore at epoch : {epoch}")
            self.print_score(precision, recall, accuracy, f1score, total_loss)


    def test(self, data):
        total_loss = 0
        predictions, truth = [], []
        for x in data:
            y_preds, loss = self.forward(x, train=True)
            y = x['chunk_tags']
            predictions += y_preds
            truth += y
            total_loss += loss
    
        total_loss /= len(data)
        precision, recall, accuracy, f1score = self.evaluate(predictions, truth)
        # print(f"\n\nTest Score is : ")
        self.print_score(precision, recall, accuracy, f1score, total_loss)
