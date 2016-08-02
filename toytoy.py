from keras.layers import Input,GRU, RepeatVector, Dense
from keras.models import Model
from keras import backend as K
import numpy as np


print(np.mean([[0,0],[0,1],[0,0],[0,0]]))

train = np.array([
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
])

test = np.array([
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
])

train = train.reshape(5,4,4)
test = test.reshape(3,4,4)

def c_objective(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)

inputs = Input(shape=(4, 4))
encoded = GRU(4, activation='hard_sigmoid', return_sequences=True)(inputs)
encoded = GRU(4, activation='hard_sigmoid', return_sequences=False)(encoded)
encoded = Dense(2, activation='relu')(encoded)
decoded = Dense(4, activation='relu')(encoded)
decoded = RepeatVector(4)(decoded)
decoded = GRU(16, activation='hard_sigmoid', return_sequences=True)(decoded)
decoded = GRU(4, activation='hard_sigmoid', return_sequences=True)(decoded)
m = Model(inputs, decoded)
m.compile(optimizer='adadelta', loss=c_objective, metrics=['accuracy'])
m.summary()
# train
m.fit(train,
      train,
      nb_epoch=5000,
      batch_size=5,
      shuffle=True,
      validation_data=(test, test))

print('test')
print(test[2])
print(m.predict(np.array([test[2]])))