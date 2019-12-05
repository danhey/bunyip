import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform

norm = True

headers = ['q', 'r1', 'r2', 'tratio', 'incl', 'ecc', 'per0']
params_frame = pd.read_csv('detached/db_database_params.dat', delimiter=' ', header=None, names=headers)
db_database = np.load('detached/db_database.npy')
x_train = db_database
y_train = params_frame.values

test_params_frame = pd.read_csv('detached/db_test_params.dat', delimiter=' ', header=None, names=headers)
test_db_database = np.load('detached/db_test.npy')

x_test = test_db_database
y_test = test_params_frame.values

#-------
# Load model
#-------

model_path = "./RELU_2000_2000_lr=1e-05_norm_insert2000layer-1571628331/NN.h5"
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model(model_path)
model.summary()

y_hat = model.predict(x_test)

if norm:
  mu = np.mean(y_train, axis = 0)
  std = np.std(y_train, axis = 0)
  y_hat = y_hat * std + mu

for i in range(len(y_hat)):
    print([ float('%.4f' % j) for j in y_hat[i]])
    print([ float('%.4f' % j) for j in y_test[i]])
    print()