import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#load dataset (https://www.kaggle.com/datasets/shahir/protein-data-set)
df = pd.read_csv('pdb_data_no_dups.csv')
df = df.dropna() #drop data entries with values of nan
df = df.sample(n=2000, random_state=42)

# keep top 5 classes
top_classes = df['classification'].value_counts().nlargest(5).index
df = df[df['classification'].isin(top_classes)]
df = df.drop(['structureId', 'pdbxDetails', 'publicationYear', 'crystallizationMethod'], axis=1)

# balance class weights
min_count = df['classification'].value_counts().min()
df_balanced = df.groupby('classification').sample(n=min_count, random_state=42)

# encode categorical features
df_encoded = pd.get_dummies(df_balanced, columns=['experimentalTechnique', 'macromoleculeType'])

# define X and Y labels
X = df_encoded.drop(['classification'], axis=1).values
Y = df_balanced['classification'].values.reshape(-1, 1)

# one hot encode y labels
Y_categories, Y_inv = np.unique(Y, return_inverse=True)
Y_onehot = np.zeros((Y.size, Y_categories.size))
Y_onehot[np.arange(Y.size), Y_inv] = 1

# normalize X labels
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split data into train/test set
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_onehot, test_size=0.2, random_state=42)

# define activation functions
def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# initialize parameters using He initialization
def init_params(input_size, h1_size, h2_size, output_size):
    np.random.seed(42)
    W1 = np.random.randn(input_size, h1_size) * np.sqrt(2. / input_size)
    b1 = np.zeros((1, h1_size))
    W2 = np.random.randn(h1_size, h2_size) * np.sqrt(2. / h1_size)
    b2 = np.zeros((1, h2_size))
    W3 = np.random.randn(h2_size, output_size) * np.sqrt(2. / h2_size)
    b3 = np.zeros((1, output_size))
    return W1, b1, W2, b2, W3, b3

# implement forward propagation
def forward_prop(X, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = softmax(Z3)
    cache = (Z1, A1, Z2, A2, Z3, A3)
    return cache, A3

# implement back propagation
def back_prop(X, Y, cache, W1, W2, W3):
    Z1, A1, Z2, A2, Z3, A3 = cache
    m = X.shape[0]

    dZ3 = A3 - Y
    dW3 = np.dot(A2.T, dZ3) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m

    dZ2 = np.dot(dZ3, W3.T) * (Z2 > 0)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dZ1 = np.dot(dZ2, W2.T) * (Z1 > 0)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # use gradient clipping to remove exploding gradients
    max_norm = 5.0
    for grad in [dW1, db1, dW2, db2, dW3, db3]:
        np.clip(grad, -max_norm, max_norm, out=grad)

    return dW1, db1, dW2, db2, dW3, db3

# train model
def train(X_train, Y_train, h1_size, h2_size, alpha, iterations):
    input_size = X_train.shape[1]
    output_size = Y_train.shape[1]
    W1, b1, W2, b2, W3, b3 = init_params(input_size, h1_size, h2_size, output_size)

    for i in range(iterations):
        cache, A3 = forward_prop(X_train, W1, b1, W2, b2, W3, b3)
        dW1, db1, dW2, db2, dW3, db3 = back_prop(X_train, Y_train, cache, W1, W2, W3)

        W1 -= alpha * dW1
        b1 -= alpha * db1
        W2 -= alpha * dW2
        b2 -= alpha * db2
        W3 -= alpha * dW3
        b3 -= alpha * db3

        if i % 100 == 0:
            loss = -np.mean(np.sum(Y_train * np.log(A3 + 1e-8), axis=1))
            print(f"Iteration {i} | Loss: {loss:.4f}")

    return W1, b1, W2, b2, W3, b3

# predict y labels
def predict(X, W1, b1, W2, b2, W3, b3):
    _, A3 = forward_prop(X, W1, b1, W2, b2, W3, b3)
    return np.argmax(A3, axis=1)

# run the model
W1, b1, W2, b2, W3, b3 = train(X_train, Y_train, h1_size=64, h2_size=32, alpha=0.0001, iterations=2000)

# evaluate test accuracy
y_pred = predict(X_test, W1, b1, W2, b2, W3, b3)
y_actual = np.argmax(Y_test, axis=1)
accuracy = np.mean(y_pred == y_actual)
print(f"Balanced Accuracy: {accuracy:.4f}")
