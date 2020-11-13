#HIT PREDICTION MODEL FOR 00'S DATASET

#Loading The Libraries

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import joblib
#########################################################################
 
#Loading The Dataset

df = pd.read_csv('H:\TechnoColabs Projects\Spotify Project\Main Code\dataset-of-60s.csv')
######################################################

#Shuffling The Data
df = df.sample(frac=1)
#(df.shape) =====> rows=8642, columns=19 
#################################################

X = df.drop(['target', 'track', 'artist', 'uri'], axis=1)
#X.shape ==> rows = 8642, cols = 15
Y = df['target']
hit_flop_count = Y.value_counts()
#hits(1)=4321; flops(0)=4321 ==========> Data is balanced
Y = Y.values
####################################################

#Standardizing The Input Data
std_scaler = StandardScaler()
std_scaled_X = std_scaler.fit_transform(X)
####################################################

#Splitting Data Into Training, Validation And Testing
x_train, x_test, y_train, y_test = train_test_split(std_scaled_X, Y, test_size=0.1, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=(1/9), random_state=1)
#x_train.shape=(4696, 15), x_test.shape=(588, 15), x_val.shape=(588, 15)
###################################################

#Deep Learning Model training Algorithm

nn = 200 #num of neurons in hidden layers
target_count = 2

model_60 = tf.keras.Sequential()
model_60.add(tf.keras.layers.Flatten())
model_60.add(tf.keras.layers.Dense(nn, activation=tf.nn.relu))# first hidden  layer
model_60.add(tf.keras.layers.Dense(nn, activation=tf.nn.relu))# second hid)den layer
model_60.add(tf.keras.layers.Dense(nn, activation=tf.nn.relu))# third hidd)en layer
model_60.add(tf.keras.layers.Dense(target_count, activation=tf.nn.softmax))# output layer
  
model_60.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_60 = model_60

fit_model = model_60.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val), batch_size=100)

predictions = model_60.predict(x_test)
###########################################################
#Visualizing Neural Network Loss History

#Loss Variation Plot
training_loss = fit_model.history['loss']
validation_loss = fit_model.history['val_loss']
epoch_count1 = range(1, len(training_loss) + 1)

plt.subplot(2,1,2)
plt.title('Loss Variation Plot')
plt.plot(epoch_count1, training_loss, color='violet', label='Training Loss')
plt.plot(epoch_count1, validation_loss, color='indigo', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

#Accuracy Variation Plot
training_acc = fit_model.history['accuracy']
validation_acc = fit_model.history['val_accuracy']
epoch_count2 = range(1, len(training_acc) + 1)

plt.subplot(2,1,1)
plt.title('Accuracy Variation Plot')
plt.plot(epoch_count2, training_acc, color='red', label='Training Accuracy')
plt.plot(epoch_count2, validation_acc, color='green', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

#Predictions Testing
print(predictions[215])
print('Predicted:', np.argmax(predictions[215]))
print('Original:', y_test[215])

tf.keras.models.save_model(model_60, 'Trained_model_60')