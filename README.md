# POS-tag-Implementation-using-Recurrent-Perceptron
This repository contains scripts and instructions for training a Single Recurrent Perceptron to identify mark Noun Chunks in a sentence. In a noun chunk, only the noun is compulsory, determiners and adjectives are optional.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```
Then run

```bash
python nltk_req_download.py
```


## Training

1. To train on the entire data use the following code:

```bash
python train.py
```

2. To train using the 5-fold cross-validation method:
```bash
python cross_val_train.py
```

**NOTE:** The model right now is using the Mean Square Error (MSE) instead of cross_entropy (CE) loss. Through experiments, we have found that MSE works best as compared to CE. Using MSE we were able to satisfy all the conditions given in the question whereas in CE one condition was failing. Searching upon the issue we came to know that CE is susceptible to noisy data, which is the case here.
