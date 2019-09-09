import tensorflow as tf 
from tensorflow import keras
from keras.models import load_model as load_model_file

import numpy as np

class CNN:
  model = None
  emission_matrix = None

  def __init__(self):
    pass

  def load_model(self,path):
    self.model = load_model_file(path)

  def load_emission_matrix(self,path):
    self.emission_matrix = np.load(path)

  def generate_emission_matrix(self,indexes,save_path):
    images,labels = getDataSet(indexes)
    bMatrix = np.zeros((4,4))
    for i in range(4):
      # For each true label, calculate predicted labels
      state_images = images[labels==i]
      preds = self.model.predict(state_images)
      for j in range(4):
        prop = np.sum(np.argmax(preds,axis=1)==j)/preds.shape[0]
        bMatrix[i,j] = prop
    np.save(save_path,bMatrix)

    def get_predictions(index):
      data = 
      return np.argmax(model)


