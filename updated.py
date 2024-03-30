import numpy as np
import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle



st.set_page_config(layout='wide')
st.markdown("<h1 style='text-align: center;'>Chunking</h1> <br><h3>Enter URL Link Below:</h3>", unsafe_allow_html=True)

class Model():
    def __init__(self, learning_rate=0.1) -> None:
        self.learning_rate = learning_rate
        self.weights = np.random.rand(11, 1)
        self.grad = np.zeros((11, 1))
        pass

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
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, F1-score: {f1score:.4f}, Loss: {loss:.4f}")

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
                # input = target + start + one_hot_prev_pos_tag + one_hot_curr_pos_tag
                # print((np.array([y]), sos, one_hot_prev_pos_tag, one_hot_curr_pos_tag, np.array([-1.0])))
                input = np.concatenate((np.array([y]), sos, one_hot_prev_pos_tag, one_hot_curr_pos_tag, np.array([-1.0])))
            else:
                # input = predicted + start + one_hot_prev_pos_tag + one_hot_curr_pos_tag
                input = np.concatenate((np.array([y_pred]), sos, one_hot_prev_pos_tag, one_hot_curr_pos_tag, np.array([-1.0]))) 

            y = chunk_tags[i]
            y_logit = np.matmul(input, self.weights)[0]
            y_pred = 1 if y_logit >= 0.5 else 0
            y_preds.append(y_pred)
            accuracy += 1 if y_pred==y else 0
            grad_y = self.weights[0] * grad_y + input.reshape(11, 1) # 11*1

            grad += (y_logit - y) * grad_y # 11x1 Sum of loss at each token
            loss += ((y_logit - y)**2)/2

        
        grad /= len(pos_tags)
        self.grad = grad
        accuracy /= len(pos_tags)
        self.save_weights()
        return y_preds, loss
    
    def backward(self):
        # print(f"{sum([i[0] for i in self.learning_rate * self.grad]):.4f} : ", end = " ")
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
            # print(f"Weights are : ", [i[0] for i in self.weights])

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
        # print(np.array(predictions).shape, np.array(truth).shape)
        precision, recall, accuracy, f1score = self.evaluate(predictions, truth)
        print(f"\n\nTest Score is : ")
        self.print_score(precision, recall, accuracy, f1score, total_loss)



if __name__ == "__main__":
    # with open('train.jsonl') as f:
    #     train_data = [json.loads(line) for line in f]

    model = Model()
    epochs = 3


    # model.train(train_data, epochs)
    # model.test(test_data)
    model.load_weights()

    # with open('test.jsonl') as f:
    #     test_data = [json.loads(line) for line in f]
    #
    # model.test(test_data)

    user_input = st.text_input("")
    caption_button = st.button("Predict")
    if caption_button:
        st.balloons()
        if user_input:
            chunks, sentence = model.inference(user_input)
            print(sentence, chunks)
            st.markdown(f"<h2 style='text-align: center;'>{sentence}</h2>",
                            unsafe_allow_html=True)
        else:
            print("Enter a sentence for doing prediction")