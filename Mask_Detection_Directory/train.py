#  create the training script called train.py in the MaskDetectionDirectory directory

import argparse
from azureml.core import Run


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# let user feed in hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--training-lr', type=float, dest='training_lr', default=1e-04, help='training learning rate')
parser.add_argument('--training-epochs', type=int, dest='training_epochs', default=10, help='training epochs')
parser.add_argument('--fine-tuning-lr', type=float, dest='fine_tuning_lr', default=1e-05, help='fine tuning learning rate')
parser.add_argument('--fine-tuning-epochs', type=int, dest='fine_tuning_epochs', default=5, help='fine tuning epochs')
args = parser.parse_args()

data_folder = args.data_folder
print('Data folder:', data_folder)



# Data augmentation
validation_split = 0.20

datagen = ImageDataGenerator(
    rescale=1.0/255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode="nearest",
    rotation_range=20,
    shear_range=0.15,
    horizontal_flip=True,
    validation_split=validation_split
)

train_generator = datagen.flow_from_directory(
    args.data_folder, 
    target_size=(128, 128),
    color_mode="rgb",
    class_mode='categorical',
    batch_size = 32,
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    args.data_folder, 
    target_size=(128, 128),
    color_mode="rgb",
    class_mode='categorical',
    batch_size = 8,
    subset='validation'
)


# get hold of the current run
run = Run.get_context()


# Transfer Learning
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(128,128,3))
base_model.trainable = False

last_layer = base_model.layers[-1]
last_output = last_layer.output

x = keras.layers.Flatten()(last_output)
x = keras.layers.Dense(128, activation = 'relu')(x)
x = keras.layers.Dense(3, activation = 'softmax')(x)
model = keras.Model(base_model.input, x)

# compile the model
model.compile(optimizer = Adam(learning_rate = args.training_lr ),
              loss = 'categorical_crossentropy',  
              metrics = ['accuracy']
             )

# fit the model
history = model.fit( train_generator,
                     epochs = args.training_epochs,
                     steps_per_epoch = train_generator.n // 32, 
                     validation_data = validation_generator,
                     verbose = 0,
                     validation_steps = validation_generator.n // 8
)


# Fine-Tuning
base_model.trainable = True

model.compile(optimizer = Adam(learning_rate = args.fine_tuning_lr ),
              loss = 'categorical_crossentropy',  
              metrics = ['accuracy']
             )

history = model.fit(train_generator,
                    epochs = args.fine_tuning_epochs,
                    steps_per_epoch = train_generator.n // 32, 
                    validation_data = validation_generator,
                    verbose = 0,
                    validation_steps = validation_generator.n // 8
)

# training and validation accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

run.log('training learning rate', np.float(args.training_lr))
run.log('training epochs', np.float(args.training_epochs))
run.log('fine tuning learning rate', np.float(args.fine_tuning_lr))
run.log('fine tuning epochs', np.float(args.fine_tuning_epochs))


run.log('accuracy', np.array(acc))
run.log('val_accuracy', np.array(val_acc))
run.log('loss', np.array(loss))
run.log('val_loss', np.array(val_loss))


os.makedirs('outputs', exist_ok=True)
# file saved in the outputs folder is automatically uploaded into experiment record
model.save("outputs/1")
model.save('outputs/my_model.h5')
