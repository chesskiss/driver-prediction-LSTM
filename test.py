# import torch
# import time
# import tensorflow as tf

# from keras.models import load_model
# from keras import layers, models
# from keras.layers import Dense, BatchNormalization, LSTM, Dropout, Input, Flatten, LayerNormalization
# from keras.models import Sequential
# import numpy as np
# import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, LayerNormalization, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from main import pre_process_encoder, performance_plot #parse_files
tf.random.set_seed(7)

LOOK_BACK   = 16
TRAIN_SIZE  = 0.67

def plot(train_history):
    # Plotting the training and validation accuracy
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(train_history.history['accuracy'], label='Train Accuracy')
    # plt.plot(train_history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(train_history.history['loss'], label='Train Loss')
    # plt.plot(train_history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.savefig('training_history.png') 
    plt.show()

def gpu():
    ############## Version 1 ##########################
    # Check if TensorFlow is using the GPU
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # Simple TensorFlow operation to test GPU usage
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)


    print("Result of matrix multiplication:\n", c)

    ############## Version 2 ##########################
    devices = tf.config.list_physical_devices()
    print("\nDevices: ", devices)

    gpus = tf.config.list_physical_devices('GPU')
    print("\nGPU: ", gpus)

    if gpus:
        details = tf.config.experimental.get_device_details(gpus[0])
        print("GPU details: ", details)


    # #check for gpu
    # if torch.backends.mps.is_available():
    #     mps_device = torch.device("mps")
    #     x = torch.ones(1, device=mps_device)
    #     print (x)
    # else:
    #     print ("MPS device not found.")


    # # GPU
    # start_time = time.time()

    # # syncrocnize time with cpu, otherwise only time for oflaoding data to gpu would be measured
    # torch.mps.synchronize()

    # a = torch.ones(4000,4000, device="mps")
    # for _ in range(200):
    #     a +=a

    # elapsed_time = time.time() - start_time
    # print( "GPU Time: ", elapsed_time)









    X = pd.DataFrame()
    y = []
    'Features and labels'
    for i, person in enumerate(files):
        for data in person:
            df = pd.DataFrame(data)

            x_sample = df.drop(columns=['datetime', 'fuel'] ).dropna()
            y_sample = [i for _ in range(len(x_sample))]

            if "Car_Id" in X.columns:
                x_sample.drop('Car_Id', axis=1, inplace=True)
            if 'Trip' in X.columns:
                x_sample.drop('Trip', axis=1, inplace=True)

            X = pd.concat([X, x_sample], ignore_index=True)
            y += y_sample

    return X,y  





def deep_lstm_model(x):
    model = Sequential(
        [
            Input(shape=(x.shape[1], LOOK_BACK)), #ragged=True
            LSTM(160, return_sequences=True),
            LayerNormalization(),
            LSTM(200, return_sequences=True),
            LayerNormalization(),
            Flatten(),
            # Dense(256, activation='relu'), # kernel_initializer='glorot_uniform', bias_initializer='zeros'
            Dense(3, activation='sigmoid') #TODO change to y_test shape
        ]
    )    
    return model

def scaling(X, y):
    df= X.copy() #TODO

    # (X - E(X))/σ(X)
    std_scale = StandardScaler()
    std_scale.fit(X)
    x = std_scale.transform(X)

    #range between (0,1)
    zero1_scaler = MinMaxScaler(feature_range=(0, 1))
    x = zero1_scaler.fit_transform(X)

    df[:]= x
    df['timestamp'] = X['timestamp']
    
    x_train, x_test,  y_train, y_test = train_test_split(df, y, train_size=TRAIN_SIZE, shuffle=False) #TODO shuffle?
    return x_train, x_test, y_train, y_test



def simple_run():

    # convert an array of values into a dataset matrix
    def create_dataset(dataset, LOOK_BACK=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-LOOK_BACK-1):
            a = dataset[i:(i+LOOK_BACK), 0]
            dataX.append(a)
            dataY.append(dataset[i + LOOK_BACK, 0])
        return np.array(dataX), np.array(dataY)
    
    x, y = pre_process_encoder()
    x_train, x_test, y_train, y_test = scaling(x, y)
    
    train = [x_train, y_train]
    print(len(train))
    trainX, trainY = create_dataset(train, LOOK_BACK)
    testX, testY = create_dataset([x_test, y_test], LOOK_BACK)
    # reshape input to be [samples, time steps, features]
    print(trainX.shape)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))




    model = deep_lstm_model(trainX)
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2) #TODO epochs=100

    scaler = MinMaxScaler(feature_range=(0, 1))
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))


    performance_plot(history)




gpu()

#TODO - very useful : https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/


def model_load():
    pass
    # Replace 'your_model.h5' with the path to your .h5 file
    # model = load_model('your_model.h5')


def delete_this():
    dataframe = pd.read_csv('test.csv', usecols=[1], engine='python')

    dataset = dataframe.values
    dataset = dataset.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    print(f'dataset size: train - {len(train)}, test - {len(test)}')