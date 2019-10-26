import numpy as np
from keras.models import load_model, Sequential
from keras.layers import Conv2D, Dense, Activation, LeakyReLU, Softmax
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy


def genmodel():
    vgg = load_model("vgg16.h5")

    new_model = Sequential()
    for layer in vgg.layers[:20]:
        layer.trainable = False
        new_model.add(layer)

    new_model.add(Dense(62))
    new_model.add(LeakyReLU(.1))
    new_model.add(Dense(20))
    new_model.add(Softmax())

    return new_model

def get_gen():
    datagen = ImageDataGenerator(
        rescale = 1/255.,


    )

    train_gen= datagen.flow_from_directory(
        "bird_dataset/bird_dataset/train_images",
        target_size = (224, 224),
        batch_size = 32,
        class_mode = "categorical")

    val_generator = datagen.flow_from_directory(

    )

    return train_gen, val_generator

def main():
    model = genmodel()
    traingen, valgen = get_gen()

    model.compile(Adam(lr=1e-4),categorical_crossentropy, metrics = ["accuracy"])

    model.fit_generator(generator = traingen,
                        steps_per_epoch = 34,
                        epochs = 10,
                        validation_data = valgen,
                        validation_steps = 4
                        )

if __name__ == '__main__':
    main()



