import os
import sys
import time
import threading
import numpy as np
from enum import Enum
import firebase_admin
from firebase_admin import credentials, db
from keras import models
from prediction_model import model_prediction

REAL_TIME_REFERANCE = 'LiveData'
OBD_REFERENCE       = 'Obd'

obd_dict = {} #{key: value} - key: obd_id, value: row_counter

def get_live_data():
    while True:
        obd_ref = db.reference(REAL_TIME_REFERANCE)
        obd_snapshot = obd_ref.get()
        print("Listening for new live data...")
        if not obd_snapshot:
            print("Didn't find new live data.")
            time.sleep(10)
            continue

        for obd_id in obd_snapshot.keys():
            if obd_id not in obd_dict:
                print(f"OBD {obd_id} started driving.")

                obd_dict[obd_id] = 0
                if isinstance(obd_snapshot[obd_id], dict):
                    obd_dict[obd_id] = int (list(obd_snapshot[obd_id].keys())[0])
                print(f'Live drives: {obd_dict.keys()}')
            
            lock = threading.Lock()
            # with lock:
                # start_time = time.time()
                # thread = threading.Thread(target=run_algorithm, args=(obd_id, start_time,))

            #     thread.start()
            start_time = time.time()
            run_algorithm(obd_id, start_time) #TODO delete


        time.sleep(5)  # Run this check every 20 seconds


def run_algorithm(obd_id, start_time):
    data =[]
    first5rows = True
    while True:
        row_index = obd_dict[obd_id]
        print(f' \nrow index = {row_index} \n') #TODO delete, only for testing
        obd_ref = db.reference(f'{REAL_TIME_REFERANCE}/{obd_id}')
        obd_snapshot = obd_ref.get()
        print(f'len of obd snap = {len(obd_snapshot)}')
        # print(list(obd_snapshot.keys())[0])
        finished = False

        new_data = []
        if isinstance(obd_snapshot, dict):
            if obd_snapshot and str(row_index) in list(obd_snapshot.keys()):
                # data.append(obd_snapshot[str(row_index)])
                for i in range(row_index, len(obd_snapshot)): 
                    new_data.append(obd_snapshot[str(i)]) 
                data.extend(new_data)
            else:
                finished = True

        elif isinstance(obd_snapshot, list):
            if obd_snapshot and len(obd_snapshot) > row_index:
                # data.append(obd_snapshot[row_index])
                for i in range(row_index, len(obd_snapshot)): 
                    new_data.append(obd_snapshot[i]) 
                data.extend(new_data)
            else:
                finished = True

        if finished:
            #del obd_dict[obd_id]
            print(f'no more data from OBD {obd_id}')
            break

        else:
            print(f'Checked OBD {obd_id} row {row_index}: {data[-1]}')


        ####################### APPLY ALGORITHM HERE ##################################

        # data- a list of dictionaries, every dictionary is a second from the driving
        # the list grows every iteration, the list is ordered
        # data[0] is the first second from the driving
        # data[-1] is the last second that never checked
        # row_index is the index of the row, but it's better to use data[-1] because row_index not always starts at 0
        # you should update driver_result to the UID of the driver that the algorithm found that he is the driver.

        ## EXAMPLES
        print(f'Current Thread = {threading.current_thread()}')

        uids        = ['2W5Nq5aZ4cP9VA6zEWBbi7FicxE2', 'lT3ip6zL8gU34vuoONy5UTmWwPg1', 'vcAN0KURuBYtNhztFCJJR9y4EhR2']
        
        # print(f'\n{len(data)}\n') #TODO delete
        obd_dict[obd_id] = len(obd_snapshot)
        
        if first5rows and len(data)>=5:
            first5rows = False
            data = data[5:]

        print('len data = ', len(data))

        prediction, data = model_prediction(uids, data)
        print(f'prediction= {prediction}')

        max_speed = max(item['speed'] for item in data)
        print(f'max speed = {max_speed}')
        if prediction and (time.time() - start_time >= 20*60 or max_speed > 30):
            db.reference(OBD_REFERENCE).child(obd_id).child('last_driver').set(prediction)
            break
        else:
            prediction = prediction if prediction != 'STOLEN' else 'UNKOWN DRIVER'
            db.reference(OBD_REFERENCE).child(obd_id).child('last_driver').set(prediction)

        #time.sleep(1)

        # if max(prediction) > 0.8: #if prediction certainty is greater than 80%
        #     return #TODO do we need to return a value? e.g. driver_result
        # #TODO change while True to a for loop after which car reported stolen (ideal), or add a "forth" driver to the model that will be the thief.
    


def run_algo_server():
    print("Start listening for new drives...")
    # threading.Thread(target=get_live_data, daemon=True).start() #TODO
    get_live_data()

if __name__ == '__main__':
    # Initialize Firebase Admin SDK
    while True:
        try:
            cred = credentials.Certificate(f"{os.getcwd()}/car-driver-bc91f-firebase-adminsdk-xhkyn-214c09b623.json")
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://car-driver-bc91f-default-rtdb.asia-southeast1.firebasedatabase.app/'
            })
            print("Firebase initialized successfully.")
            break
        except Exception as e:
            print(f"Error initializing Firebase: {e}. Retrying in 5 seconds...")
            time.sleep(1)

    run_algo_server()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Server stopped by user")


def run_algorithm(obd_id):

            

        # מעדכנים את מונה השורות
        obd_dict[obd_id] += len(new_data)
        print(f'Collected {len(new_data)} new rows for OBD {obd_id}')