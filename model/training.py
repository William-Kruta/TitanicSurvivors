import numpy as np
import pandas as pd

from model.utils import one_hot_encode, create_label_mapping, min_max_scaler
from model.model import Model


def read_file():
    df = pd.read_csv("./dataset/titanic.csv")
    return df


def get_training_data(train_size: float = 0.8):

    df = read_file()

    df.drop("Cabin", axis=1, inplace=True)
    df.dropna(inplace=True)
    # Encode string features
    df["Embarked"] = create_label_mapping(df["Embarked"])
    df["Sex"] = create_label_mapping(df["Sex"])
    # Scale numerical features
    df["Age"] = min_max_scaler(df["Age"])
    df["Fare"] = min_max_scaler(df["Fare"])
    feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    target_col = "Survived"
    # Split data into training and testing sets
    train_size = int(len(df) * train_size)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    # Split data into features and target
    train_X, train_y = train[feature_cols], train[target_col]
    test_X, test_y = test[feature_cols], test[target_col]
    return train_X, train_y, test_X, test_y


def train_model(epochs: int, eval_on_finish: bool = True):
    train_X, train_y, test_X, test_y = get_training_data()
    print(train_X)
    train_y = train_y.values.reshape(-1, 1)
    model = Model(input_size=len(train_X.columns), hidden_size=24)

    model.train(train_X.values, train_y, epochs=epochs, learning_rate=0.01)

    if eval_on_finish:
        predict(model, test_X, test_y)


def predict(model, test_X, test_y):
    df = test_X.copy()
    pred = model.predict(test_X)
    pred.columns = ["prediction"]
    # Add to dataframe
    df = pd.concat([df, pred], axis=1)
    df = pd.concat([df, test_y], axis=1)
    # Determine number of correct predictions
    df["correct"] = df["prediction"] == df["Survived"]
    num_true = (df["correct"] == True).sum()
    accuracy = (num_true / len(df)) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    df.to_csv("predictions.csv", index=False)
