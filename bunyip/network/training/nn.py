import numpy as np
import pandas as pd
import tensorflow.keras as keras
import os
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import CustomObjectScope

norm = True

# Set paths
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
main_path = "/run/user/1001/gvfs/smb-share:server=silo.physics.usyd.edu.au,share=silo4/snert/alison/"
results_folder = main_path + "dan/networks/"
logs_folder = results_folder +  "logs/"

# load the data
headers = ['q', 'r1', 'r2', 'tratio', 'incl', 'ecc', 'per0']
params_frame = pd.read_csv('detached/db_database_params.dat', delimiter=' ', header=None, names=headers)
db_database = np.load('detached/db_database.npy')

x_train = db_database
y_train = params_frame.values

test_params_frame = pd.read_csv('detached/db_test_params.dat', delimiter=' ', header=None, names=headers)
test_db_database = np.load('detached/db_test.npy')

x_test = test_db_database
y_test = test_params_frame.values

if norm:
  mu = np.mean(y_train, axis = 0)
  std = np.std(y_train, axis = 0)

  y_train = (y_train - mu)/std
  y_test = (y_test - mu)/std

# ------------
#   NETWORK
# ------------

neurons1 = 2000
lr = 1e-5
ID = "RELU_{}_{}_lr={}".format(neurons1, neurons1, lr)
if norm:
  ID += "_norm"

# ------------
#   SET UP TENSORBOARD
# ------------

NAME = ID + "-{}".format(int(time.time()))

results_folder += NAME + "/"

if os.path.exists(results_folder):
    shutil.rmtree(results_folder)
os.makedirs(results_folder)
print("RESULTS FOLDER:", results_folder)

print('set up tensorboard')
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience = 500, factor=0.2, min_lr=1e-9)

tb = keras.callbacks.TensorBoard(log_dir=logs_folder + '{}'.format(NAME),
                                 histogram_freq = False,
                                 write_graph = False,
                                 write_images= False)

cp_callback_path = results_folder + "/cp-{epoch:03d}.h5"
cp = keras.callbacks.ModelCheckpoint(cp_callback_path,
                                              verbose=1,
                                              save_weights_only=False,
                                              period=10)

callbacks = [tb, reduce_lr]

# ------------
#   CREATE NETWORK
# ------------

# THIS IS THE ORIGINAL MODEL
# model = keras.Sequential()

# model.add(keras.layers.Dense(neurons1,
#                              activation='relu',
#                              input_shape = (100,),
#                              ))

# model.add(keras.layers.Dense(neurons1,
#                              activation='relu',
#                              ))

# model.add(keras.layers.Dense(7,
#                              activation='linear',
#                              ))

# THIS LOADS AN EXISTING MODEL TO CONTINUE TRAINING IT
model_name = "RELU_2000_2000_lr=1e-05_norm_insert2000layer-1571628331/"
model_path = main_path + "dan/networks/{}/NN.h5".format(model_name)
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model(model_path)

# THIS REMOVES THE LAST LAYER AND THEN ADDS NEW LAYERS TO MAKE THE NETWORK LARGER
# WE ALSO STOP ANY OF THE EXISTING LAYERS (ALREADY TRAINED) FROM CONTINUING TO BE TRAINED
model.pop()
model.layers[0].trainable = False
model.layers[1].trainable = False
model.layers[2].trainable = False
model.layers[3].trainable = False

model.add(keras.layers.Dense(2000,
                             activation='relu',
                             name = "dense_5"
                             ))

model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(7,
                             activation='linear',
                             name = "dense_6"))
model.summary()


# ------------
#   COMPILE AND FIT THE NETWORK
# ------------

model.compile(optimizer = keras.optimizers.Adam(lr=lr),
              loss = 'mse')


model.fit(x_train, y_train,
  epochs=2000,
  validation_data = (x_test, y_test), callbacks = callbacks)

with open(results_folder + "/model_info.txt", 'w') as f:
    for k, v in sorted(model.__dict__.items()):
        if k[0] != '_':
            f.write(str(k) + str(v) + "\n")
    model.summary(print_fn=lambda x: f.write(x + '\n'))

model.save(results_folder + '/NN.h5')

print("RESULTS FOLDER:", results_folder)
