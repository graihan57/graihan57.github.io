---
layout: post
title: "Future-frame-prediction-using-LSTM"
author: Molla Hafizur Rahman
categories: Supervised learning
tags: [Prediction, LSTM, PCA]
Date: 2021-12-25 10:12
---

```python
import os
import glob
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
%config Completer.use_jedi = False
import tensorflow as tf
from keras.callbacks import EarlyStopping,Callback,ModelCheckpoint
from tensorflow.keras.layers import LSTM,Dense,Dropout,Input,RepeatVector,TimeDistributed
from tensorflow.keras.models import Sequential,Model
from time import time
from sklearn.metrics import mean_squared_error
```

    Using TensorFlow backend.


# Load image


```python
data_dir = r'/jet/home/mhrahman/Projects/HW7/Image Sequences'
img_file = sorted(glob.glob(os.path.join(data_dir,'*.jpg')))
```


```python
plt.figure(figsize=(20,20))
for i in range(6):
    if i >= 5:
        break
    img_ = Image.open(img_file[i]).resize((800,600))
    plt.subplot(1,6,i+1)
    plt.imshow(img_,cmap='gray',interpolation='none')
plt.savefig(r'/jet/home/mhrahman/Projects/HW7/Figures/Images.jpg',dpi = 300)
plt.show()
```


![image-center](/images/LSTM_PCA/output_3_0.png){: .align-center}{: width="650"}      



```python
img_array = np.array(Image.open(img_file[0]).resize((80,60)).convert('L'))/255
pca_img = PCA(20).fit(img_array)
```


```python
pca_img
```




    PCA(n_components=20)




```python
trans_pca = pca_img.transform(img_array)
img_arr = pca_img.inverse_transform(trans_pca)
plt.imshow(img_arr,cmap='Greys_r')
```




    <matplotlib.image.AxesImage at 0x7eff50248fd0>




![image-center](/images/LSTM_PCA/output_6_1.png){: .align-center}{: width="650" }     




```python
trans_pca.shape
```




    (60, 20)




```python
trans_pca.reshape(1,-1).shape[1]
```




    1200



# PCA Analysis and Flattening


```python
all_array = []
for i in range(0, len(img_file)):
    img_ar = np.array(Image.open(img_file[i]).resize((80,60)).convert('L'))/255
    pca_ = PCA(20).fit(img_ar)
    pca_image = pca_.transform(img_ar)
    pca_flatten = pca_image.reshape(1,-1)
    all_array.append(pca_flatten[0])
```


```python
i_array = np.array(all_array)
i_array.shape
```




    (11, 1200)



# Split the data for input and output


```python
def split_data(data,step_in, step_out):
    X, Y = list(), list()
    for i in range(len(data)):
        end_ix = i + step_in
        out_end_ix = end_ix + step_out

        if out_end_ix > len(data):
            break
        seq_x, seq_y = data[i:end_ix,:],data[end_ix : out_end_ix,:]
        X.append(seq_x)
        Y.append(seq_y)
    return np.array(X), np.array(Y)
```


```python
def split_file(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # Finding the end of the pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out

        # Checking if we are beyond the sequence
        if out_end_ix > len(sequence):
            break

        # Gather input and output parts of pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix : out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return X,y
```


```python
step_in = 25
step_out = 25
```


```python
X, Y = split_data(i_array,step_in,step_out)
```


```python
file_x, file_y = split_file(img_file,step_in,step_out)
```

# Save and load the data


```python
np.save("/ocean/projects/mch210006p/mhrahman/HW7/X.npy", X)
np.save("/ocean/projects/mch210006p/mhrahman/HW7/Y.npy", Y)
```


```python
X = np.load("/ocean/projects/mch210006p/mhrahman/HW7/X.npy")
Y = np.load("/ocean/projects/mch210006p/mhrahman/HW7/Y.npy")
```


```python
X.shape
```




    (9952, 25, 1200)



# Split for training and testing


```python
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,shuffle = False)
```


```python
x_train_file,x_test_file,y_train_file,y_test_file = train_test_split(file_x,file_y,test_size = 0.2,shuffle = False)
```


```python
n_features = x_train.shape[2]
```

# Model Building


```python
model_LSTM = Sequential([
    LSTM(100,return_sequences=True,input_shape = (x_train.shape[1],x_train.shape[2]),activation = 'relu'),
    Dropout(0.2),
    TimeDistributed(Dense(n_features,activation = 'linear'))
])
```


```python
n_past = step_in
n_future = step_out
```


```python
encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
encoder_l1 = tf.keras.layers.LSTM(100, return_state=True)
encoder_outputs1 = encoder_l1(encoder_inputs)

encoder_states1 = encoder_outputs1[1:]
decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs1[0])
decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
decoder_outputs1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l1)
model_e1d1 = tf.keras.models.Model(encoder_inputs,decoder_outputs1)
```


```python
model = model_LSTM
model.summary()
with open('modelsummary_LSTM.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm_1 (LSTM)                (None, 25, 100)           520400    
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 25, 100)           0         
    _________________________________________________________________
    time_distributed (TimeDistri (None, 25, 1200)          121200    
    =================================================================
    Total params: 641,600
    Trainable params: 641,600
    Non-trainable params: 0
    _________________________________________________________________



```python
class TimeCallback(Callback):
    def on_train_begin(self,logs={}):
        self.logs=[]
    def on_epoch_begin(self,epoch,logs={}):
        self.starttime = time()
    def on_epoch_end(self,epoch,logs={}):
        self.logs.append(time()-self.starttime)
es = EarlyStopping(monitor='val_loss',mode='min',verbose=1, patience = 20,min_delta = 1)
cb = TimeCallback()
checkpoints = ModelCheckpoint('weight_01.hdf5',monitor='val_loss',verbose=1,save_best_only= True,mode='min')
```


```python
model.compile(optimizer='adam', loss = 'mean_squared_error')
```


```python
epochs = 30
batch = 32
history = model.fit(x_train,y_train,epochs=epochs,
                    batch_size = batch,validation_split= .2,verbose = 1,
                   shuffle = False, callbacks = [es,cb,checkpoints])
```

    Train on 6272 samples, validate on 1569 samples
    Epoch 1/2
    6240/6272 [============================>.] - ETA: 0s - loss: 37.8017
    Epoch 00001: val_loss improved from inf to 56.18157, saving model to weight_03.hdf5
    6272/6272 [==============================] - 59s 9ms/sample - loss: 37.6360 - val_loss: 56.1816
    Epoch 2/2
    4064/6272 [==================>...........] - ETA: 14s - loss: 19.0049

# Plot training and validation loss


```python
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.plot(epochs,loss,'r')
plt.plot(epochs,val_loss,'b')
plt.title('Training and validation loss')
plt.xlabel('epochs',fontsize = 12)
plt.ylabel('Loss', fontsize = 12)
plt.legend(["Training loss","Validation loss"])
plt.savefig(r'/jet/home/mhrahman/Projects/HW7/Figures/Loss.jpg', dpi = 300)
plt.show()
```


```python
plt.plot(cb.logs)
plt.title('Time per epoch')
plt.xlabel('Epoch')
plt.ylabel('Time')
plt.legend(['Time'],loc = 'upper right')
#path = r'/jet/home/mhrahman/Projects/HW1/Figures/Classification_loss.jpg'
plt.savefig(r'/jet/home/mhrahman/Projects/HW7/Figures/Time.jpg', dpi = 300)
plt.show()
```


```python
model = tf.keras.models.load_model('weight_03.hdf5')
```


```python
img_array = np.array(Image.open(y_test_file[200][24]).resize((80,60)).convert('L'))/255
pca_img = PCA(20).fit(img_array)
trans_pca = pca_img.transform(img_array)
plt.imshow(img_array,cmap='Greys_r')
plt.savefig(r'/jet/home/mhrahman/Projects/HW7/Figures/from_file.jpg',dpi = 300)
```


![image-center](/images/LSTM_PCA/output_38_0.png){: .align-center}{: width="650" }      




```python
imh = pca_img.inverse_transform(y_test[200][24].reshape(60,20))
plt.imshow(imh,cmap='Greys_r')
plt.savefig(r'/jet/home/mhrahman/Projects/HW7/Figures/actual.jpg',dpi = 300)
```


![image-center](/images/LSTM_PCA/output_39_0.png){: .align-center}{: width="650" }     




```python
y_pred = model.predict(x_test)
```


```python
imp = pca_img.inverse_transform(y_pred[200][24].reshape(60,20))
plt.imshow(imp,cmap='Greys_r')
plt.savefig(r'/jet/home/mhrahman/Projects/HW7/Figures/predicted.jpg',dpi = 300)
```


![image-center](/images/LSTM_PCA/output_41_0.png){: .align-center}{: width="650" }     



# Actual vs Predicted reconstructed image


```python
actual = []
predicted = []
for i in range(0,500,50):
    img_array = np.array(Image.open(y_test_file[i][24]).resize((80,60)).convert('L'))/255
    pca_img = PCA(20).fit(img_array)
    ima = pca_img.inverse_transform(y_test[i][24].reshape(60,20))
    imp = pca_img.inverse_transform(y_pred[i][24].reshape(60,20))
    actual.append(ima)
    predicted.append(imp)
```


```python
plt.figure(figsize=(20,20))
for i in range(6):
    if i >= 5:
        break
    img_ = actual[i]
    plt.subplot(1,6,i+1)
    plt.imshow(img_,cmap='gray',interpolation='none')
plt.savefig(r'/jet/home/mhrahman/Projects/HW7/Figures/Ac_images.jpg',dpi = 300)
plt.show()
```


![image-center](/images/LSTM_PCA/output_44_0.png){: .align-center}{: width="650" }    




```python
plt.figure(figsize=(20,20))
for i in range(6):
    if i >= 5:
        break
    img_ = predicted[i]
    plt.subplot(1,6,i+1)
    plt.imshow(img_,cmap='gray',interpolation='none')
plt.savefig(r'/jet/home/mhrahman/Projects/HW7/Figures/Pr_images.jpg',dpi = 300)
plt.show()
```


![image-center](/images/LSTM_PCA/output_45_0.png){: .align-center}{: width="650" }    



# Average MSE


```python
error = []
for i in range(len(y_pred)):
    err = mean_squared_error(y_pred[i][24],y_test[i][24])
    error.append(err)

final_error = np.mean(error)
print(final_error)
```

    0.07973217082965939
