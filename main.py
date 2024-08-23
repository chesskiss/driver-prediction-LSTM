import os
from enum import Enum
import seaborn as sns
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
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, LSTM, Dropout, Input, Flatten, LayerNormalization
from keras import layers
from keras import models as m
from keras.optimizers import Adam
tf.random.set_seed(0)



def performance_plot(train_history, driver):
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
    plt.savefig('training_history_'+driver+'.png') 
    plt.show()

def plot_data(x,y):
    df = x
    df['driver'] = y
    features = x.columns # List of features to plot
    df['time'] = range(1, len(df) + 1)

    # Create a FacetGrid with 6 plots (one for each feature)
    g = sns.FacetGrid(x.melt(id_vars=['time', 'driver'], value_vars=features), 
                    col="variable", hue="driver", col_wrap=3, height=4)

    # Map the scatterplot to each facet
    g.map(sns.scatterplot, "time", "value", alpha=0.7)


    g.add_legend()
    g.set_axis_labels("Time", "Feature Value")
    g.set_titles(col_template="{col_name}")

    plt.savefig('dataset_facet_grid_plot.png', dpi=300, bbox_inches='tight')
    plt.show()



def download_firebase_train():
    local_path = f"/Users/arnoldcheskis/Documents/Projects/Archive/LimudNaim/Driving_project_lesson-LimudNaim/data/"

    'Initialize Firebase Admin SDK'
    cred = credentials.Certificate(f"{os.getcwd()}/car-driver-bc91f-firebase-adminsdk-xhkyn-214c09b623.json")
    firebase_admin.initialize_app(cred, {'storageBucket': 'car-driver-bc91f.appspot.com'})

    bucket  = storage.bucket()
    for driver in Drivers:
        dir = "drives/" + driver.value
        blobs   = bucket.list_blobs(prefix=dir)
        os.makedirs(local_path + str(driver.value))
        for i, blob in enumerate(blobs):
            local = local_path + str(driver.value) + '/' + str(i) + '.csv'
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
        id = os.path.basename(data_dir) #person's ID
        csv_data.append((person, id))
    return csv_data

def pre_process_encoder(files, driver):
    X = pd.DataFrame()
    y = []
    'Features and labels'
    for person, id in files:
        for data in person:
            df = pd.DataFrame(data)

            x_sample = df.drop(columns=['datetime', 'fuel', 'speedLimit', 'timestamp'] ).dropna()
            y_sample = [1 if id == driver else 0 for _ in range(len(x_sample))]

            if "Car_Id" in X.columns:
                x_sample.drop('Car_Id', axis=1, inplace=True)
            if 'Trip' in X.columns:
                x_sample.drop('Trip', axis=1, inplace=True)

            X = pd.concat([X, x_sample], ignore_index=True)
            y += y_sample


    return X,y  



'Split the data set into window samples'
def window(X1, y1):
    X_samples = []
    y_samples = []
    
    # encoder = LabelEncoder() #TODO used to convert string, or other labels to integers
    # encoder.fit(y1)
    # y1 = encoder.transform(y1)
    
    Xt = np.array(X1)
    yt= np.array(y1).reshape(-1,1)

    overlapsize = T//2
    n = yt.size    

    # for over the 263920  310495 in jumps of 64
    for i in range(0, n , T-overlapsize):
        # grab from i to i+length
        sample_x = Xt[i:i+T,:]
        if sample_x.shape[0] == T: 
            X_samples.append(sample_x)

        sample_y = yt[i:i+T]
        if sample_y.shape[0] == T: 
            y_samples.append(sample_y)

    return np.array(X_samples),  np.array(y_samples)


def rnn_dimension(X,y, train_size):
    # (X - E(X))/σ(X)
    std_scale = StandardScaler()
    std_scale.fit(X)
    x = std_scale.transform(X)

    #range between (0,1)
    zero1_scaler = MinMaxScaler(feature_range=(0, 1))
    x = zero1_scaler.fit_transform(X)


    # plot_data(pd.DataFrame(x),y) #TODO 


    X_samples, y_samples = window(x, y)
    y_samples_cat = to_categorical(np.array(y_samples))
    
    X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn = train_test_split(X_samples, y_samples_cat, train_size=train_size, shuffle=True) #TODO shuffle?
    # X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn = train_test_split(x, to_categorical(np.array(y)), train_size=train_size, shuffle=True) #TODO shuffle?
    
    return X_train_rnn, y_train_rnn, X_test_rnn, y_test_rnn



def deep_model():
    model = Sequential(
        [
            Input(shape=(T, PROPS)), #ragged=True
            # LSTM(64, return_sequences=True),
            Dense(120, activation='relu'),
            LayerNormalization(),
            # LSTM(128, return_sequences=False),
            Dense(200, activation='relu'),
            LayerNormalization(),
            # Flatten(),
            # Dense(256, activation='relu'), # kernel_initializer='glorot_uniform', bias_initializer='zeros'
            Dense(2, activation='sigmoid')
        ]
    )    
    return model



def prediction(models, x):
    predicitons = []
    for model, id in models:
        predicitons.append((model.predict(x)[-1,-1,1], id))
    predicitons = np.array(predicitons)
    values = predicitons[:,0].astype(float)
    return 'CAR STOLEN!' if np.max(values) < 0.85 else predicitons[np.argmax(values),1]




PROPS       = 6 #TODO X.shape[1] #Properties of the drivers
T           = 16
LR          = 0.1
EPOCHS      = 10
TRAIN_SIZE  = 0.85
BATCH_SIZE  = T



class Drivers(Enum):
    d1 = '2W5Nq5aZ4cP9VA6zEWBbi7FicxE2'
    d2 = 'lT3ip6zL8gU34vuoONy5UTmWwPg1'
    d3 = 'vcAN0KURuBYtNhztFCJJR9y4EhR2'


def main():
    # download_firebase_train() #Call only once to get contents from Firebase
    files = parse_files() #extract objects from files
    

    #TODO return acceleration and train again
    testx = []
    testy = None
    label = []

    models = []
    for driver in Drivers:
        X, y = pre_process_encoder(files, driver.value) #get raw data in pd.Frame
        X_train, y_train, X_test, y_test  = rnn_dimension(X, y, TRAIN_SIZE)
        testx = X_test[0:1]
        label = y_test[0:1]
        testy = driver.value

        model_file = 'Model_' + driver.value + '.keras'
        if os.path.isfile(model_file):
            models.append((m.load_model(model_file), driver.value))
        # if True: #TODO try LSTM again
        else:         # TODO turn to a function ? 
            model = deep_model()
            # model = model()
            optimizer = Adam(learning_rate=LR) #TODO add clipvalue=1.0 ?
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy']) #loss
            model_history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
            model.save(model_file)
            models.append(model)
            performance_plot(model_history, driver)


    #TODO eval accuracy error for combined model function
    
    #TODO check if without LSTM we only need 16 lines? I will look at the last one anyway. Do we need to give it more than 1 window??
    # for model in models:
    #     score = model.evaluate(X_test, y_test) # batch_size=50
    #     print('Test loss:', score[0])
    #     print('Test accuracy:', score[1])

    #     print(testy[0].shape)
    #     print(f'output after training = \n{model.predict(testx[0:1])[0,-1]} \n\n {model.predict(testx[1:2])[0,-1]}')
    #     print(f'\n y test = {testy[0]} , {testy[1]}')
    
    print(testx[0:1].shape)
    print(label[0,-1])
    print(prediction(models, testx[0:1]))
    # plot_data(X,y)



if __name__ == '__main__':
    main()
