import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import keras
import sklearn
warnings.filterwarnings("ignore")



train_df = pd.read_csv("Data/mitbih_train.csv",header=None)
test_df = pd.read_csv("Data/mitbih_test.csv",header=None)

train_df[187]=train_df[187].astype(int)
equal = train_df[187].value_counts()
print(equal)

df_1 = train_df[train_df[187]==1]
df_2 = train_df[train_df[187]==2]
df_3 = train_df[train_df[187]==3]
df_4 = train_df[train_df[187]==4]
df_0 = (train_df[train_df[187]==0]).sample(n=20000,random_state=42)

df_1_upsample=sklearn.utils.resample(df_1,replace=True,n_samples=20000,random_state=123)
df_2_upsample=sklearn.utils.resample(df_2,replace=True,n_samples=20000,random_state=124)
df_3_upsample=sklearn.utils.resample(df_3,replace=True,n_samples=20000,random_state=125)
df_4_upsample=sklearn.utils.resample(df_4,replace=True,n_samples=20000,random_state=126)

train_df=pd.concat([df_0,df_1_upsample,df_2_upsample,df_3_upsample,df_4_upsample])
equal = train_df[187].value_counts()
print(equal)

c = train_df.groupby(187,group_keys=False).apply(lambda train_df:train_df.sample(1))
print(c)

def add_gaussian_noise(signal):
    noise=np.random.normal(0,0.5,186)
    return(signal+noise)

tempo=c.iloc[0,:186]
bruiter = add_gaussian_noise(tempo)
'''
plt.subplot(2,1,1)
plt.plot(c.iloc[0,:186])
plt.subplot(2,1,2)
plt.plot(bruiter)
plt.show()
'''

target_train=train_df[187]
target_test=test_df[187]
y_train=keras.utils.to_categorical(target_train)
y_test=keras.utils.to_categorical(target_test)
X_train=train_df.iloc[:,:186].values
X_test=test_df.iloc[:,:186].values
for i in range(len(X_train)):
    X_train[i,:186]= add_gaussian_noise(X_train[i,:186])
X_train = X_train.reshape(len(X_train), X_train.shape[1],1)
X_test = X_test.reshape(len(X_test), X_test.shape[1],1)


def network(X_train,y_train,X_test,y_test):
    im_shape=(X_train.shape[1],1)
    inputs_cnn=keras.layers.Input(shape=(im_shape), name='inputs_cnn')
    conv1_1=keras.layers.Convolution1D(64, (6), activation='relu', input_shape=im_shape)(inputs_cnn)
    conv1_1=keras.layers.BatchNormalization()(conv1_1)
    pool1=keras.layers.MaxPool1D(pool_size=(3), strides=(2), padding="same")(conv1_1)
    conv2_1=keras.layers.Convolution1D(64, (3), activation='relu', input_shape=im_shape)(pool1)
    conv2_1=keras.layers.BatchNormalization()(conv2_1)
    pool2=keras.layers.MaxPool1D(pool_size=(2), strides=(2), padding="same")(conv2_1)
    conv3_1=keras.layers.Convolution1D(64, (3), activation='relu', input_shape=im_shape)(pool2)
    conv3_1=keras.layers.BatchNormalization()(conv3_1)
    pool3=keras.layers.MaxPool1D(pool_size=(2), strides=(2), padding="same")(conv3_1)
    flatten=keras.layers.Flatten()(pool3)
    dense_end1 = keras.layers.Dense(64, activation='relu')(flatten)
    dense_end2 = keras.layers.Dense(32, activation='relu')(dense_end1)
    main_output = keras.layers.Dense(5, activation='softmax', name='main_output')(dense_end2)
    
    
    model = keras.models.Model(inputs= inputs_cnn, outputs=main_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics = ['accuracy'])
    
    
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=8),
             keras.callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    history=model.fit(X_train, y_train,epochs=40,callbacks=callbacks, batch_size=32,validation_data=(X_test,y_test))
    model.load_weights('best_model.h5')
    return(model,history)


model,history=network(X_train,y_train,X_test,y_test)

