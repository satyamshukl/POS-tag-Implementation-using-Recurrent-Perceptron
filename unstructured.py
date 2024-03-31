import numpy as np
import json

import pickle

# def preprocess(data):
#     for line in data:
#         line['token'] = ['^'] + line['token']

class SingleRecurrentPerceptron:
    def __init__(self, learning_rate=0.1) -> None:
        self.learning_rate = learning_rate
        self.weights = np.random.rand(11, 1)
        pass


    def loss(self, predicted, target):
        """
        Predicted is a list (for all timesteps)
        target is also a list
        """
        loss = 0
        for idx, val in enumerate(predicted):
            loss += (((target[idx] - val) ** 2)/2)
        loss = loss / len(predicted)

        return loss
    
            
    
    def create_one_hot(self, n, max=4):
        x = np.zeros(max)
        x[n - 1] = 1

        return x

    

    def train(self, data, train=True, threshold=0.0):
        total_loss = 0.0
        acc = 0.0
        for p, example in enumerate(data):
            # print(f'\nExample {p}')
            pos_tags = example["pos_tags"]
            chunk_tags = example["chunk_tags"]
            target = np.array([0])
            predicted = np.array([0])
            start = np.array([1])
            logits_list = []   
            example_loss = 0
            acc_temp = 0.0
            grad_y = np.zeros((11, 1))
            grad = np.zeros((11, 1))
            for i, pos_tag in enumerate(pos_tags):
                # print('Time step ', i)        
                if i > 0:
                    start = [0]
                    one_hot_prev_pos_tag = self.create_one_hot(pos_tags[i-1])
                    # one_hot_prev_pos_tag[3] = 1 
                else:
                    one_hot_prev_pos_tag = np.zeros(4)

                one_hot_curr_pos_tag = self.create_one_hot(pos_tag)
                # print(f'target: {target}')
                # print(f'target shape: {target.shape}')


                if train:
                    # input = target + start + one_hot_prev_pos_tag + one_hot_curr_pos_tag
                    input = np.concatenate((target, start, one_hot_prev_pos_tag, one_hot_curr_pos_tag, np.array([-1.0])))
                else:
                    # input = predicted + start + one_hot_prev_pos_tag + one_hot_curr_pos_tag
                    input = np.concatenate((predicted, start, one_hot_prev_pos_tag, one_hot_curr_pos_tag, np.array([-1.0])))                

                target = np.array([chunk_tags[i]])
                logits = np.matmul(input, self.weights)
                predicted = self.predict(logits)
                acc_temp += (1 if (int(predicted[0]) == int(target[0])) else 0)

                grad_y = self.weights[0] * grad_y + input.reshape(11, 1) # 11*1

                grad += (logits - target) * grad_y # 11x1 Sum of loss at each token

                # logits_list.append(logits[0])
                example_loss += ((logits[0]-target[0])**2)/2
                # example_loss += self.loss(logits_list, chunk_tags)

            grad /= len(pos_tags)
            acc_temp /= len(pos_tags)
            acc += acc_temp

            self.backward(grad)

            example_loss /= len(pos_tags)
            total_loss += example_loss
            # total_loss += self.loss(logits_list, chunk_tags)


        total_loss /= len(data)
        acc /= len(data)

        return total_loss, acc


    def predict(self, logits, threshold=0.5):
        # print('Logits: ', logits)
        if float(logits[0]) >= threshold:
            predicted = [1]
        else:
            predicted = [0]

        return predicted
    

    def backward(self, grad):
        self.weights -= self.learning_rate * grad

            

if __name__ == "__main__":
    with open('train.jsonl') as f:
        data = [json.loads(line) for line in f]

    model = SingleRecurrentPerceptron()
    epochs = 5

    for epoch in range(epochs):
        loss, acc = model.train(data)
        print(f"Epoch {epoch}/{epochs}\tLoss {loss}\tAccuracy {acc}")

    print('Weights: ', model.weights)







