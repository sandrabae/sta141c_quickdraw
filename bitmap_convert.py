###Converts numpy bitmaps into usable test and training datasets for processing
###in Keras.

import numpy as np
import pickle


files = ["aircraft_carrier","airplane","apple","bowtie","candle"
         ,"door","octagon","pants","parachute","pencil"]

for file in files:
    filename = file + ".npy"
    pre_image = np.load(filename)

    samples = np.shape(pre_image)[0]

    split = int(samples*0.8)

    pre_image_train = pre_image[0:split]
    pre_image_test = pre_image[split:samples]

    image_train = []
    image_test = []

    for image in pre_image_train:
        image_train.append(np.reshape(image,(28,28)))

    for image in pre_image_test:
        image_test.append(np.reshape(image,(28,28)))

    train_filename = file + ".train"
    test_filename = file + ".test"

    train_file = open(train_filename, mode = 'wb')
    test_file = open(test_filename, mode = 'wb')

    pickle.dump(image_train, train_file)
    pickle.dump(image_test, test_file)
