
import os
import pandas as pd 
import numpy as np 

import tensorflow as tf 
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense


class CNN:
  """
  This class trains a CNN and saves the resulting model files to
  the specified path
  """
  def __init__(self,image_path,label_path,save_path):
    """
    Initializes the class. Needs 3 paths.

    image_path : string
      Path to folder with images
    label_path : string
      Path to folder with labels
    save_path : string
      Path to folder where to save model
    """
    self.image_path = image_path
    self.save_path = save_path
    self.label_path = label_path
    test_frac = 0.1
    validation_frac = 0.1
    self.getDataIndex(test_frac,validation_frac)

    self.test_images,self.test_labels = self.load_dataset(self.test_ind)
    self.train_images,self.train_labels = self.load_dataset(self.train_ind)
    self.val_images,self.val_labels = self.load_dataset(self.val_ind)

    self.model = self.getModel()
    self.datagen = self.getDataGen()

  def save_files(self):
    """

    Saves all files

    """

    # Indexes
    train_ind_path = os.path.join(self.save_path,'train_ind.npy')
    np.save(train_ind_path,self.train_ind)
    test_ind_path = os.path.join(self.save_path,'test_ind.npy')
    np.save(train_ind_path,self.test_ind)
    val_ind_path = os.path.join(self.save_path,'val_ind.npy')
    np.save(train_ind_path,self.val_ind)

    # Model
    model_save_path = os.path.join(self.save_path,'model.h5')
    self.model.save(model_save_path)

    # Model History
    val_loss = self.model.history.history['val_loss']
    train_loss = self.model.history.history['loss']

    loss_path = os.path.join(self.save_path,'loss.npy')
    val_loss_path = os.path.join(self.save_path,'val_loss.npy')

    np.save(loss_path,train_loss)
    np.save(loss_path,val_loss)

  
  def train(self,epochs):
    """
    Train the neural network

    Trains for epochs epochs
    """
    batch_size = 16
    train_generator = self.datagen.flow(self.train_images,
      self.train_labels,batch_size) 
    validation_generator = self.datagen.flow(self.val_images,
      self.val_labels,batch_size)

    self.model.fit_generator(train_generator,
      steps_per_epoch=2000//batch_size,
      epochs = epochs,
      validation_data=validation_generator,
      validation_steps = 800//batch_size)

  def load_dataset(self,ind):
    """
    This function loads the datasets
    """
    dataSet = np.empty((1,32,32,1))
    dataLabels = np.empty(1)
    for i in ind:
      data = np.load(os.path.join(self.image_path,'cell{}.npy'.format(int(i))))
      label = np.load(os.path.join(self.label_path,'labels{}.npy'.format(int(i))))
      data = data/np.max(data)
      print(data.shape)
      data = np.reshape(data,(data.shape[0],32,32,1))
      dataSet = np.append(dataSet,data,axis=0) 
      dataLabels = np.append(dataLabels,label,axis=0)
    return dataSet[1:,:,:,:],dataLabels[1:]


  def getDataIndex(self,test_frac,validation_frac):
    """
    Splits the data indexes into a test set, a validation set
    and a training set according to the specified fractions
    """
    ids_path = os.path.join(image_path,'details.pkl')
    ids_df = pd.read_pickle(ids_path)
    labeled = os.path.join(label_path,'indexes.npy')
    labels = np.load(labeled)
    ids = ids_df['ID']
    ids = np.array(list(set(ids).intersection(set(labels))))
    num_traces = ids.size

    test_size = int(num_traces*test_frac)
    val_size = int(num_traces*validation_frac)

    test_ind = np.random.choice(ids,test_size,replace=False)
    test_ind = test_ind.astype('int')
    self.test_ind = test_ind

    remaining_ind = np.array(list(set(ids)-set(test_ind)))

    val_ind = np.random.choice(remaining_ind,val_size,replace=False)
    val_ind = val_ind.astype('int')
    self.val_ind = val_ind

    train_ind = np.array(list(set(remaining_ind)-set(val_ind)))
    train_ind.astype('int')
    self.train_ind = train_ind




  def getModel(self):
    """
    Builds the model
    """
    model = Sequential()
    model.add(Conv2D(32,(3,3),input_shape=(32,32,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',
      metrics =['accuracy','categorical_accuracy'])
    return model

  def getDataGen(self):
    """ Defines the datagenerator
    """
    datagen = ImageDataGenerator(
      rotation_range=40,
      zoom_range=0.2,
      horizontal_flip = True,
      vertical_flip = True,
      fill_mode = 'nearest')
    return datagen

if __name__ == '__main__':
  EPOCHS = 2
  # Load current path
  PATH,filename = os.path.split(os.path.realpath(__file__))
  label_path = os.path.join(PATH,'Labels/')
  image_path = os.path.join(PATH,'Images/')
  save_path = os.path.join(PATH,'ModelFit/')

  cnn = CNN(image_path,label_path,save_path)
  cnn.train(EPOCHS)
  cnn.save_files()



