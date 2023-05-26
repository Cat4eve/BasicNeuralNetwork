import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv('hospital_deaths_train.csv')
target = 'In-hospital_death'

X = df.drop([target], axis=1).to_numpy()

imp = SimpleImputer(missing_values=np.nan, strategy='median')
imp.fit(X)
X = imp.transform(X)

y = df[target].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(X.shape[1]),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
])

loss = tf.keras.losses.BinaryCrossentropy()

model.compile(
    optimizer='adam',
    loss=loss,
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=5)

pred = []
for i in model.predict(X_test):
    if i < 0.5:
        pred.append(0)
    else:
        pred.append(1)

print(accuracy_score(pred, y_test))


from DenseLayer import DenseLayer
import numpy as np

class DenseNetwork:
    def __init__(self, layers: list[DenseLayer]=None, iteration_number=10000, learning_rate=0.01):
        self.layers = layers
        self.iter = iteration_number  # iterations number for gradient descent
        self.lr = learning_rate  # learning rate for gradient descent
        for i in range(len(self.layers) - 1):
            self.layers[i+1].previous_layer = self.layers[i]
            self.layers[i].next_layer = self.layers[i+1]

    def call(self, X, y, training=True):
        if training:
            print(self.layers[-1].call(X))
            for _ in range(self.iter):
                for l in range(len(self.layers)):
                    for i in range(len(self.layers[l].weights)):
                        for j in range(len(self.layers[l].weights[i])):
                            d = self.RSS2(self.layers[-1].call(X), y, self.layers[l], i, j)
                            self.layers[l].weights[i][j] = self.layers[l].weights[i][j] - self.lr*d
                            # print(l.weights[i][j])
                # print(self.RSS(self.layers[-1].call(X), y))
            self.call(X, y, False)
        else:
            print(self.layers[-1].call(X))
            print(self.RSS(self.layers[-1].call(X), y))

    def RSS2(self, pred, actual, layer, i, j):
        return sum(pred-actual) * layer.get_weight_d(i, j)

    def RSS(self, pred, actual):
        return sum((pred-actual)*(pred-actual))


nn = DenseNetwork(layers=[
    DenseLayer(2,3),
    DenseLayer(3,4)
])

nn.call(
    np.array([1,5]),
    np.array([2,7,20,5]),
    True
)