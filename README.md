# Titanic Survival Prediction using Neural Networks

### Introduction

This project is a custom implementation of a binary classification model using neural networks to identify the likelyhood of a passenger surviving the sinking of the Titanic.

This is accomplished by using information about the passenger. Such as their age, gender, class, and other features.

The neural network was made using Numpy. No frameworks such as Keras, PyTorch, or TensorFlow were used.

The dataset can be found here: https://www.kaggle.com/datasets/brendan45774/test-file?select=tested.csv

### Features

Below is a sample of the input features used for the network.

```
Pclass = Passenger Class
Sex = Gender of the passenger
Age = Age of the passenger
SibSp = Number of siblings/spouses aboard the Titanic
Parch = Number of parents/children aboard the Titanic
Fare = Fare paid for the ticket
Embarked = Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)  # Note: These appear as integers in the training data.


  Pclass  Sex       Age  SibSp  Parch      Fare  Embarked
     3    1  0.452723      0      0  0.015282         1
     3    0  0.617566      1      0  0.013663         2
     2    1  0.815377      0      0  0.018909         1
     3    1  0.353818      0      0  0.016908         2
     3    0  0.287881      1      1  0.023984         2
..      ...  ...       ...    ...    ...       ...       ...
     2    0  0.630753      0      2  0.071731         2
     1    1  0.512066      0      0  0.057971         0
     3    0  0.248319      1      1  0.030726         0
     3    1  0.353818      0      0  0.015412         2
     1    1  0.393380      0      0  0.050749         2

```

### Evaluation

- This is the result after running the model for **1000** epochs:

```
Epoch 0, Loss: 0.6932
...
Epoch 1000, Loss: 0.6290

Accuracy: 58.21%
```

---

- This is the result after running the model for **1500** epochs:

```
Epoch 0, Loss: 0.6931
...
Epoch 1500, Loss: 0.3862

Accuracy: 85.07%
```

---

- This is the result after running the model for **2000** epochs:

```
Epoch 0, Loss: 0.6930
...
Epoch 2000, Loss: 0.1534

Accuracy: 100.00%
```
