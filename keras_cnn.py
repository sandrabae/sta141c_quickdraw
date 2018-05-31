
import keras
from keras.modelskeras.m  import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')


#Convolutional Neural Network (CNN) with Keras.
#I will use a model from the tutorial by Jason Brownlee. It has the following 9 layers:

'''
    1. Convolutional layer with 30 feature maps of size 5×5.
    2. Pooling layer taking the max over 2*2 patches.
    3. Convolutional layer with 15 feature maps of size 3×3.
    4. Pooling layer taking the max over 2*2 patches.
    5. Dropout layer with a probability of 20%.
    6. Flatten layer.
    7. Fully connected layer with 128 neurons and rectifier activation.
    8. Fully connected layer with 50 neurons and rectifier activation.
    9. Output layer.
'''

# reshape to be [samples][pixels][width][height]
X_train_cnn = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test_cnn = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# define the CNN model
def cnn_model():
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#getting the timing 

    np.random.seed(0)
    # build the model# build
    model_cnn = cnn_model()
    # Fit the model
    model_cnn.fit(X_train_cnn, y_train_cnn, validation_data=(X_test_cnn, y_test_cnn), epochs=10, batch_size=200)
    # Final evaluation of the model
    scores = model_cnn.evaluate(X_test_cnn, y_test_cnn, verbose=0)
    print('Final CNN accuracy: ', scores[1])
