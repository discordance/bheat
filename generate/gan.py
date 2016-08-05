from keras.optimizers import Adam
from keras.layers import Input, Dense, Reshape, LeakyReLU, Dropout, BatchNormalization, GRU, RepeatVector
from keras.models import Model
import sys
from sklearn.cross_validation import train_test_split
import numpy as np
import os
from sklearn.cross_validation import train_test_split
from tqdm import tqdm
# local package
import utils
import random


STYLE = 'latin'
FNULL = open(os.devnull, 'w')
ROOT = os.path.dirname(os.path.realpath(__file__))

beats = np.load(ROOT + '/cache/'+STYLE+'_c.npz')
beats = beats['arr_0']
x_train = beats
# x_train, x_test, _, _ = train_test_split(beats, beats, test_size=0.25, random_state=0)
# print('size', x_train.shape, x_test.shape)



def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

shp = x_train.shape[1:]
dropout_rate = 0.25
opt = Adam(lr=1e-4)
dopt = Adam(lr=1e-3)

# Build Generative model ...
g_input = Input(shape=(8,))
H = Dense(x_train.shape[1]*x_train.shape[2], activation='relu')(g_input)
H = BatchNormalization(mode=2)(H)
H = Reshape((x_train.shape[1], 20))(H)
H = GRU(40, return_sequences=True, activation='relu')(H)
g_V = GRU(20, return_sequences=True, activation='sigmoid')(H)
# H = Dense(64, activation='relu')(g_input)
# H = Dense(1024, activation='relu')(H)
# H = Dense(x_train.shape[1]*x_train.shape[2], activation='relu')(H)
# H = Dense(x_train.shape[1]*x_train.shape[2], activation='hard_sigmoid')(H)
# #H = Dropout(dropout_rate)(H)
# g_V = Reshape((x_train.shape[1], 20))(H)

generator = Model(g_input,g_V)
generator.compile(loss='binary_crossentropy', optimizer='adadelta')
generator.summary()

# Build Discriminative model ...
d_input = Input(shape=shp)
H = Reshape((x_train.shape[1]*x_train.shape[2],))(d_input)
H = LeakyReLU(0.2)(H)
H = Dense(256)(H)
H = Dropout(dropout_rate)(H)
d_V = Dense(2, activation='softmax')(H)
discriminator = Model(d_input, d_V)
discriminator.compile(loss='categorical_crossentropy', optimizer='adadelta')
discriminator.summary()

make_trainable(discriminator, False)

# Build stacked GAN model
gan_input = Input(shape=(8,))
H = generator(gan_input)
gan_V = discriminator(H)
GAN = Model(gan_input, gan_V)
GAN.compile(loss='categorical_crossentropy', optimizer='adadelta')
GAN.summary()

ntrain = 10000
trainidx = random.sample(range(0,x_train.shape[0]), ntrain)
XT = x_train[trainidx,:,:]

# Pre-train the discriminator network ...
noise_gen = np.random.uniform(-1,1,size=(XT.shape[0],8))
generated_noise = generator.predict(noise_gen)
X = np.concatenate((XT, generated_noise))
n = XT.shape[0]
y = np.zeros([2*n,2])
y[:n,1] = 1
y[n:,0] = 1

make_trainable(discriminator,True)
discriminator.fit(X,y, nb_epoch=2, batch_size=128)
# accuracy
y_hat = discriminator.predict(X)
y_hat_idx = np.argmax(y_hat,axis=1)
y_idx = np.argmax(y,axis=1)
diff = y_idx-y_hat_idx
n_tot = y.shape[0]
n_rig = (diff==0).sum()
acc = n_rig*100.0/n_tot
print "Accuracy: %0.02f pct (%d of %d) right"%(acc, n_rig, n_tot)

# set up loss storage vector
losses = {"d":[], "g":[]}


# Set up our main training loop
def train_for_n(nb_epoch=5000, BATCH_SIZE=32):
    for e in tqdm(range(nb_epoch)):

        # Make generative beats
        beat_batch = x_train[np.random.randint(0, x_train.shape[0], size=BATCH_SIZE), :, :]
        noise_gen = np.random.uniform(-1, 1, size=(BATCH_SIZE, 8))
        generated_beats = generator.predict(noise_gen)

        # Train discriminator on generated images
        X = np.concatenate((beat_batch, generated_beats))
        y = np.zeros([2 * BATCH_SIZE, 2])
        y[0:BATCH_SIZE, 1] = 1
        y[BATCH_SIZE:, 0] = 1

        #make_trainable(discriminator, True)
        d_loss = discriminator.train_on_batch(X, y)
        losses["d"].append(d_loss)

        # train Generator-Discriminator stack on input noise to non-generated output class
        noise_tr = np.random.uniform(-1, 1, size=(BATCH_SIZE,8))
        y2 = np.zeros([BATCH_SIZE, 2])
        y2[:, 1] = 1

        #make_trainable(discriminator, False)
        g_loss = GAN.train_on_batch(noise_tr, y2)
        losses["g"].append(g_loss)
        tqdm.write("D loss %f, G loss %f" % (losses["d"][-1], losses["g"][-1]))

train_for_n(nb_epoch=200, BATCH_SIZE=128)
print(losses['d'][-1], losses['g'][-1])

# test
noise = np.random.uniform(-1, 1, size=[10, 8])
print(noise)
generateds = generator.predict(noise)
for generated in generateds:
    print(utils.draw(utils.clean_ml_out(generated)))