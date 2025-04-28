import tensorflow as tf 
import pandas as pd
import numpy
from tensorflow import keras
from tensorflow.keras import layers 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("C:\\Downloads\\diabetes.csv")

X = df.drop('Outcome',axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

model = keras.Sequential([
    layers.Dense(16,activation='relu',input_shape=(X_train.shape[1],)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer = "adam",
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

history = model.fit(
    X_train,y_train,
    epochs = 100,
    batch_size = 32,
    validation_split = 0.2,
    verbose=1
)

test_loss, test_accuracy = model.evaluate(X_test,y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
