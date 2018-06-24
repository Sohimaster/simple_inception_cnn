from keras.datasets import cifar10, mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Input, concatenate, AveragePooling2D
from keras.utils import np_utils
from keras.models import Model
from keras.optimizers import rmsprop
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator

def split_data(data_name):
  if data_name == 'mnist':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
    input_img = Input(shape = (28, 28, 1))
  elif data_name == 'cifar10':
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    input_img = Input(shape = (32, 32, 3))
    
  X_train = X_train / 255.0
  X_test = X_test / 255.0
  y_train = np_utils.to_categorical(y_train)
  y_test = np_utils.to_categorical(y_test)
    
  return X_train, y_train, X_test, y_test, input_img

# learning_rate scheduler
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 4:
        lrate = 0.0005
    elif epoch > 6:
        lrate = 0.0003        
    return lrate
  
def create_inception_modules(data, num):
    for i in range(num):
        tower_0 = Conv2D(32, (1, 1), padding='same', activation='relu')(data)
        tower_1 = Conv2D(64, (1,1), padding='same', activation='relu')(data)
        tower_1 = Conv2D(64, (3,3), padding='same', activation='relu')(tower_1)

        tower_2 = Conv2D(128, (1,1), padding='same', activation='relu')(data)
        tower_2 = Conv2D(128, (5,5), padding='same', activation='relu')(tower_2)

        tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(data)
        tower_3 = Conv2D(256, (1,1), padding='same', activation='relu')(tower_3)
        data = concatenate([tower_0, tower_1, tower_2, tower_3], axis=3)
    return data

def create_network(input_img):
  output = create_inception_modules(input_img, 2)
  output = AveragePooling2D((2, 2))(output)
  output = Flatten()(output)
  output = Dense(10, activation='softmax')(output)
  return output

def get_trained_model(data_name, epochs, batch_size):
  X_train, y_train, X_test, y_test, input_img = split_data(data_name)  
  output = create_network(input_img) 
  model = Model(inputs = input_img, outputs = output)

  # image augmentation
  datagen = ImageDataGenerator(
      rotation_range=15,
      width_shift_range=0.1,
      height_shift_range=0.1,
      horizontal_flip=True,
      )
  datagen.fit(X_train)

  model.compile(
      loss='categorical_crossentropy', 
      optimizer=rmsprop(lr=0.001,decay=1e-6), 
      metrics=['accuracy'])

  model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),\
                      steps_per_epoch=X_train.shape[0] // batch_size,epochs=epochs,\
                      verbose=1,validation_data=(X_test,y_test),
                      callbacks=[LearningRateScheduler(lr_schedule)])
  model.fit(
            X_train, y_train, 
            validation_data=(X_test, y_test), 
            epochs=epochs, 
            batch_size=batch_size,
            callbacks=[LearningRateScheduler(lr_schedule)])
  scores = model.evaluate(X_test, y_test, verbose=0)
  return model, scores

# to change dataset replace 'mnist' to 'cifar10'
model, scores = get_trained_model('mnist', 8, 64)

print("Accuracy: %.2f%%" % (scores[1]*100))