import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib as mpl
mpl.style.use('ggplot')

from sklearn.preprocessing import StandardScaler
import firebase_admin
from firebase_admin import credentials, storage

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, LSTM, Dropout, Input, Flatten, LayerNormalization
from keras import layers, models
from keras.optimizers import Adam


devices = tf.config.list_physical_devices()
print("\nDevices: ", devices)

gpus = tf.config.list_physical_devices('GPU')
print("\nGPU: ", gpus)

if gpus:
  details = tf.config.experimental.get_device_details(gpus[0])
  print("GPU details: ", details)

  

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


def download_firebase_train():
    storage_dir = [f"drives/2W5Nq5aZ4cP9VA6zEWBbi7FicxE2/", f"drives/lT3ip6zL8gU34vuoONy5UTmWwPg1", f"drives/vcAN0KURuBYtNhztFCJJR9y4EhR2"] # Add new directories (people) here
    local_path = f"/Users/arnoldcheskis/Documents/Projects/Archive/LimudNaim/Driving_project_lesson-LimudNaim/data/"

    'Initialize Firebase Admin SDK'
    cred = credentials.Certificate(f"{os.getcwd()}/car-driver-bc91f-firebase-adminsdk-xhkyn-214c09b623.json")
    firebase_admin.initialize_app(cred, {'storageBucket': 'car-driver-bc91f.appspot.com'})

    bucket  = storage.bucket()
    for dir_num, dir in enumerate(storage_dir):
        blobs   = bucket.list_blobs(prefix=dir)
        os.makedirs(local_path + str(dir_num))
        for i, blob in enumerate(blobs):
            local = local_path + str(dir_num) + '/' + str(i) + '.csv'
            blob.download_to_filename(local)


def parse_files():
    csv_data    = []
    current_dir = os.getcwd() + '/data'
    for dir in os.listdir(current_dir):
        data_dir    = os.path.join(current_dir, dir)
        person      = []       

        for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(data_dir, filename)
                with open(file_path, 'r') as csvfile:
                    if os.path.getsize(file_path) > 0:
                        df = pd.read_csv(csvfile)
                        person.append(df)
        csv_data.append(person)
    return csv_data


def pre_process_encoder(files):
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
'''
    X = np.array([])
    y = []
    'Features and labels'
    for i, person in enumerate(files):
        for data in person:
            df = pd.DataFrame(data)

            x_sample = df.drop(columns=['datetime', 'fuel'] ).dropna()
            y_sample = [i for _ in range(len(x_sample))]

            if "Car_Id" in x_sample.columns:
                x_sample.drop('Car_Id', axis=1, inplace=True)
            if 'Trip' in x_sample.columns:
                x_sample.drop('Trip', axis=1, inplace=True)
            
            x_sample, y_sample = window(x_sample,y_sample)
            X = np.concatenate([X, x_sample[1:]])
            y += y_sample

'''

'Split the data set into window samples'
def window(X1, y1):
    X_samples = []
    y_samples = []
    
     
    encoder = LabelEncoder()
    encoder.fit(y1)
    y1 = encoder.transform(y1)
    
    overlapsize = T//2
    n = y1.size    
    
    Xt = np.array(X1)
    yt= np.array(y1).reshape(-1,1)
    
    # for over the 263920  310495in jumps of 64
    for i in range(0, n , T-overlapsize):
        # grab from i to i+length
        sample_x = Xt[i:i+T,:]
        if (np.array(sample_x).shape[0]) == T: 
            X_samples.append(sample_x)

        sample_y = yt[i:i+T]
        
        if (np.array(sample_y).shape[0]) == T: #ARC 
            y_samples.append(sample_y)  #ARC

    return np.array(X_samples),  np.array(y_samples)


'for the label Select the maximum occuring value in the given array'
def max_occuring_label(sample):
    values, counts = np.unique(sample, return_counts=True)
    ind = np.argmax(counts)
    
    return values[ind] 


'Creating y_sample label by taking only the maximum'
def label_y(y_value): #TODO - maybe reducing accuracy in overlapping label buckets by erasing some
    y_samples_1 = []
    for i in range(len(y_value)):
        y_samples_1.append(max_occuring_label(y_value[i]))
        
    return np.array( y_samples_1 ).reshape(-1,1) 


def rnn_dimension(X,y, train_size):
    X_samples, y_samples = window(X, y)
    # y_samples =  label_y(y_samples)  #TODO delete

    #Shuffling 
    # X_samples,  y_samples = shuffle(X_samples, y_samples) #TODO undo ?

    # to catagory
    y_samples_cat = to_categorical(np.array(y_samples))


    X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn = train_test_split(X_samples, y_samples_cat, train_size=train_size, shuffle=False)
    # X_train,  y_train = shuffle(X_train_rnn, y_train_rnn)
    
    # return X_train, y_train, X_test_rnn, y_test_rnn
    return X_train_rnn, y_train_rnn, X_test_rnn, y_test_rnn


def normalizing_2d(X):              
           
            scale = StandardScaler()
            scale.fit(X)

            X = scale.transform(X)
            
            return X
        


def mlp_model(y_test):
    mlp = Sequential()
    #TODO Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
    #TODO mlp.add(Input(shape=(16,)))
    mlp.add(Dense(160, input_dim=X_train.shape[1], activation='relu'))
    mlp.add(layers.BatchNormalization())
    mlp.add(layers.Dropout(0.5))
    mlp.add(Dense(120, activation='relu'))
    mlp.add(layers.BatchNormalization())
    mlp.add(Dense(y_test.shape[1], activation='softmax'))
    return mlp



def deep_lstm_model(y_test):
    model = Sequential(
        [
            Input(shape=(T, PROPS)), #ragged=True
            LSTM(512, return_sequences=True),
            LayerNormalization(),
            LSTM(512, return_sequences=True),
            LayerNormalization(),
            # Flatten(),
            # Dense(256, activation='relu'), # kernel_initializer='glorot_uniform', bias_initializer='zeros'
            Dense(y_test.shape[1], activation='softmax')
        ]
    )    
    return model




T           = 16
LR          = 0.1
EPOCHS      = 3
PROPS       = 8 #Properties of the drivers
TRAIN_SIZE  = 0.85

def main():
    # download_firebase_train() #Call only once
    files = parse_files()
    X, y = pre_process_encoder(files)


    # X_train, X_test, y_train, y_test =train_test_split(X, y, train_size=0.85,shuffle=False)


    # X_train,  y_train = shuffle(X_train, y_train)


    X_train_5, y_train_5, X_test_5,y_test_5 = rnn_dimension(X,y, TRAIN_SIZE)


    y_dummy = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_dummy, train_size=0.85, shuffle=False)

    X_train_scaled = np.copy(normalizing_2d(X_train)) #TODO RuntimeWarning: invalid value encountered

    X_train, X_test_, y_train, y_test_ = train_test_split(X_train_scaled, y_train, train_size=0.99, shuffle=True)

    

    # mask_n = np.array([not np.array_equal(label, [0.0, 0.0, 1.0]) for label in y_train])
    # mask = np.array([np.array_equal(label, [0.0, 0.0, 1.0]) for label in y_train])
    # x_filtered = X_train[mask_n]
    # y_filtered = y_train[mask_n]

    # x_filtered_test = X_train[mask]
    # y_filtered_test = y_train[mask]


    model_file = 'mlp_model.keras'
    if os.path.isfile(model_file):
        mlp = models.load_model(model_file)
    if True:
    # else:
        mlp = deep_lstm_model(y_test)
        optimizer = Adam(learning_rate=LR, clipvalue=1.0)
        mlp.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        mlp_history = mlp.fit(X_train_5, y_train_5, epochs=EPOCHS, batch_size=T)
        mlp.save(model_file)
        plot(mlp_history)




    # TODO turn to a function
    X_test_normalized = normalizing_2d(X_test_)
    score = mlp.evaluate(X_test_5, y_test_5) # batch_size=50

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    #TODO ! - next time- merge data to make SEQUENTIAL - the model should know it's related!
    df = [[pd.read_csv('test.csv')]]
    user_props, _ = pre_process_encoder(df)
    user = [1 for _ in range(len(user_props)-1)]
    x, y, _, _ = rnn_dimension(user_props,user, 1)
    print(f'output after training = {mlp.predict(x)[-1]} \n {mlp.predict(x)[0]}')

    # print("actual values : \n", y_test_5[0:25])


    # print('X_test = \n', X_test_5[0:)



    # count1 = sum(np.array_equal(element, [0.0, 0.0, 1.0]) for element in y_train)
    # count2 = sum(np.array_equal(element, [1.0, 0.0, 0.0]) for element in y_train)
    # count3 = sum(np.array_equal(element, [0.0, 1.0, 0.0]) for element in y_train)

    # print(count1)
    # print(count2)
    # print(count3)


    #TODO test - pre-process a few rows from excel and see if they are predicted accurately


    #TODO - 

    ''' TODO
    Plan - 
    1. Leave only 1 person - simplest case, so the accuracy would be ~ 1
    2. After it works - add a person
    3. train on very simple data to see the model works as we expect.

    *. evaluate at each epoch and compare test performance based on the # of epochs

    Z. model training - EPOCHS > 100, etc.
    '''

if __name__ == '__main__':
    main()
