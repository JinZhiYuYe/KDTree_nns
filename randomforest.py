# -*- coding=utf-8 -*- 
# time = '2020/6/4 17:59'

from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score

def load_dataset():
    """加载数据集"""
    np.set_printoptions(suppress=True)
    dataset = np.loadtxt(r"data.csv", delimiter=",", skiprows=1, usecols=np.arange(1, 32), dtype=str)

    # print(datase)
    # df = pd.read_csv(r"./dataset/data.csv").iloc[0:, 1:-1]
    # dataset = df.to_numpy()
    # print(dataset)
    train_y, train_x = np.split(dataset, (1,), axis=1)
    train_y = np.array([0 if i == 'M' else 1 for i in train_y.ravel()])

    train_x, val_x = np.split(train_x, (300, ), axis=0)
    train_y, val_y = np.split(train_y, (300, ), axis=0)
    return train_x.astype("float"), train_y.astype("int"), val_x.astype("float"), val_y.astype("int")


train_x, train_y, val_x, val_y = load_dataset()


model = RandomForestClassifier(max_depth=6, n_estimators=10, criterion='entropy')
model.fit(train_x, train_y)
y_pred = model.predict(val_x)
print(accuracy_score(val_y, y_pred))
