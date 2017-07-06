import numpy as np
import cv2
import tensorflow as tf
import h5py
from imagenet_classes import class_names
from tensorflow.contrib.keras.api.keras.models import Sequential,Model
from tensorflow.contrib.keras.api.keras.layers import Conv2D,ZeroPadding2D,Dense,Activation,Input,MaxPool2D,BatchNormalization,AveragePooling2D,Flatten
from tensorflow.contrib.keras.api.keras.optimizers import SGD
from tensorflow.contrib.keras.api.keras.losses import binary_crossentropy
from tensorflow.contrib.keras.python.keras import layers


# from keras.layers import Merge,merge
# from keras.models import Model
weight_path="resnet50_weights_tf_dim_ordering_tf_kernels.h5"
data_image_path="laska.png"
data_image=cv2.imread(data_image_path)
data_reshaped_image=cv2.resize(data_image,(224,224)).astype(np.float32)
print(data_reshaped_image.shape)
data_reshaped_image[:,:,0] -= 103.939
data_reshaped_image[:,:,1] -= 116.779
data_reshaped_image[:,:,2] -= 123.68
data_reshaped_imageim = data_reshaped_image.transpose((2, 0, 1))
im = np.expand_dims(data_reshaped_image, axis=0)

class ResNet50_Keras_Scratch():
    def __init__(self):

        in_=Input(shape=(224,224,3))

        # Layer 1:
        self.model=Conv2D(64,(7,7),strides=(2,2),padding="SAME")(in_)
        self.model=BatchNormalization()(self.model)
        self.model=Activation("relu")(self.model)
        self.model = MaxPool2D((3, 3), (2, 2),padding="VALID")(self.model)
        print(self.model.get_shape())

        # Layer 2:
        #   *Block 1:
        in_block=self.model
        self.model=Conv2D(64,(1,1),(1,1),padding="SAME")(self.model)
        self.model=BatchNormalization()(self.model)
        self.model=Activation("relu")(self.model)
        self.model=Conv2D(64,(3,3),(1,1),padding="SAME")(self.model)
        self.model=BatchNormalization()(self.model)
        self.model=Activation("relu")(self.model)
        self.model=Conv2D(256,(1,1),(1,1),padding="SAME")(self.model)
        self.model=BatchNormalization()(self.model)
        shortcut=Conv2D(256,(1,1),(1,1),padding="SAME")(in_block)
        shortcut=BatchNormalization()(shortcut)
        self.model=layers.add([self.model,shortcut])
        self.model=Activation("relu")(self.model)
        print(self.model.get_shape())

        #   *Block 2:
        in_block = self.model
        self.model = Conv2D(64, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(64, (3, 3), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(256, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        shortcut = in_block
        self.model = layers.add([self.model, shortcut])
        self.model = Activation("relu")(self.model)
        print(self.model.get_shape())

        #   *Block 3:
        in_block = self.model
        self.model = Conv2D(64, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(64, (3, 3), (2, 2), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(256, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        shortcut=MaxPool2D((1,1),(2,2),padding="VALID")(in_block)
        self.model = layers.add([self.model, shortcut])
        self.model = Activation("relu")(self.model)
        print(self.model.get_shape())

        # Layer 3:
        #   *Block 1:
        in_block = self.model
        self.model = Conv2D(128, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(128, (3, 3), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(512, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        shortcut = Conv2D(512,(1,1),(1,1),padding="SAME")(in_block)
        shortcut = BatchNormalization()(shortcut)
        self.model = layers.add([self.model, shortcut])
        self.model = Activation("relu")(self.model)
        print(self.model.get_shape())

        #   *Block 2:
        in_block = self.model
        self.model = Conv2D(128, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(128, (3, 3), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(512, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        shortcut = in_block
        self.model = layers.add([self.model, shortcut])
        self.model = Activation("relu")(self.model)
        print(self.model.get_shape())

        #   *Block 3:
        in_block = self.model
        self.model = Conv2D(128, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(128, (3, 3), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(512, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        shortcut = in_block
        self.model = layers.add([self.model, shortcut])
        self.model = Activation("relu")(self.model)
        print(self.model.get_shape())

        #   *Block 4:
        in_block = self.model
        self.model = Conv2D(128, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(128, (3, 3), (2, 2), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(512, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        shortcut = MaxPool2D((1,1),(2,2),padding="VALID")(in_block)
        self.model = layers.add([self.model, shortcut])
        self.model = Activation("relu")(self.model)
        print(self.model.get_shape())

        # Layer 4:
        #   *Block 1:
        in_block = self.model
        self.model = Conv2D(256, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(256, (3, 3), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(1024, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        shortcut = Conv2D(1024,(1,1), (1, 1), padding="SAME")(in_block)
        shortcut = BatchNormalization()(shortcut)
        self.model = layers.add([self.model, shortcut])
        self.model = Activation("relu")(self.model)
        print(self.model.get_shape())

        #   *Block 2:
        in_block = self.model
        self.model = Conv2D(256, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(256, (3, 3), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(1024, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        shortcut = in_block
        self.model = layers.add([self.model, shortcut])
        self.model = Activation("relu")(self.model)
        print(self.model.get_shape())

        #   *Block 3:
        in_block = self.model
        self.model = Conv2D(256, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(256, (3, 3), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(1024, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        shortcut = in_block
        self.model = layers.add([self.model, shortcut])
        self.model = Activation("relu")(self.model)
        print(self.model.get_shape())

        #   *Block 4:
        in_block = self.model
        self.model = Conv2D(256, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(256, (3, 3), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(1024, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        shortcut = in_block
        self.model = layers.add([self.model, shortcut])
        self.model = Activation("relu")(self.model)
        print(self.model.get_shape())

        #   *Block 5:
        in_block = self.model
        self.model = Conv2D(256, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(256, (3, 3), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(1024, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        shortcut = in_block
        self.model = layers.add([self.model, shortcut])
        self.model = Activation("relu")(self.model)
        print(self.model.get_shape())

        #   *Block 6:
        in_block = self.model
        self.model = Conv2D(256, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(256, (3, 3), (2, 2), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(1024, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        shortcut = MaxPool2D((1,1),(2,2),padding="VALID")(in_block)
        self.model = layers.add([self.model, shortcut])
        self.model = Activation("relu")(self.model)
        print(self.model.get_shape())

        # Layer 5:
        #   *Block 1:
        in_block = self.model
        self.model = Conv2D(512, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(512, (3, 3), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(2048, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        shortcut = Conv2D(2048,(1,1),(1,1),padding="SAME")(in_block)
        shortcut = BatchNormalization()(shortcut)
        self.model = layers.add([self.model, shortcut])
        self.   model = Activation("relu")(self.model)
        print(self.model.get_shape())

        #   *Block 2:
        in_block = self.model
        self.model = Conv2D(512, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(512, (3, 3), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(2048, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        shortcut = in_block
        self.model = layers.add([self.model, shortcut])
        self.model = Activation("relu")(self.model)
        print(self.model.get_shape())

        #   *Block 3:
        in_block = self.model
        self.model = Conv2D(512, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(512, (3, 3), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation("relu")(self.model)
        self.model = Conv2D(2048, (1, 1), (1, 1), padding="SAME")(self.model)
        self.model = BatchNormalization()(self.model)
        shortcut = in_block
        self.model = layers.add([self.model, shortcut])
        self.model = Activation("relu")(self.model)
        print(self.model.get_shape())

        self.model=AveragePooling2D((7,7))(self.model)
        print(self.model)

        self.model=Flatten()(self.model)
        self.model=Dense(1000,activation="softmax")(self.model)

        print(self.model.get_shape())

        self.model=Model(inputs=in_,outputs=self.model)
        self.model.summary()

    def train(self):
        self.optimizer=SGD()
        # self.loss_function=binary_crossentropy()
        print("Begin Model ...")
        self.model.compile(loss="binary_crossentropy",optimizer=self.optimizer)
        # TRAIN DATA:
        # If you want to train from scratch, or fine-tuning => them can modify in this part
        # self.model.fit(x_train,y_train,batch_size=128,epochs=40,verbose=1)


        # LOAD WEIGHTS:
        self.model.load_weights(weight_path)

model=ResNet50_Keras_Scratch()
model.train()
print( class_names[np.argmax(model.model.predict(im))])
