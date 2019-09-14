import os
from discreteHMM import discreteHMM
import pandas as pd
import sys
import numpy as np
import tensorflow as tf 
from tensorflow import keras
from keras.models import load_model
import warnings

class cellCyclePredictor:
  """ cellCyclePredictor

      This class implements the cellCyclePrediction
      methods 2 and 3. In short it reads the images 
      from a "cellX.npy" file, scales them to have 
      a maximum value of one and then feeds each image
      through a CNN (or other model implemented in a 
      keras model.h5 file). The predictions are then 
      used as inputs in a HMM. 

      In method 2 the emission probabilities are updated 
      by the EM-algorithm.

      Im metod 3 only transition and initial probabilities
      are used.

      Case 2 enforces the prior that only one cell cycle
      starting in G1 is present in the cell file.

      Case 1 is a general case.

      PARAMS
      ------
      path_to_images : string
        the file path to the folder containing the 
        cellX.npy files
      path_to_parameters : string
        the file path to the folder containing the
        model.h5 file and the emissionMatrix.npy file
      path_to_save : string
        the file path to the folder in which to save
        the predictions
      hmm : hmm
        A instance of the discreteHMM class
      df : pandas DataFrame
        the dataframe to save the predictions in
      model : Keras Model
        The model used to make predictions
      emissionMatrix : numpy array
        The matrix containing the emission
        probabilities for the HMM


  """

  def __init__(self,path_to_images,path_to_parameters,path_to_save):
    self.path_to_images = path_to_images
    self.path_to_parameters = path_to_parameters
    self.path_to_save = path_to_save
    self.image_info_file = os.path.join(path_to_images,'details.pkl')
    self.hmm = discreteHMM()
    self.df = pd.DataFrame(columns = ['Object No','Start Time','State','Length','Std'])
    self.loadParameters()

  def getObservations(self,images):
    """
    Takes an image file and returns predictions from neural network

    PARAMS
    ------
    images : Tx32 x 32 numpy array
      The input images

    RETURNS
    -------
    obs : T numpy array
      The resulting predictions
    """
    T,X,Y = images.shape
    images = np.reshape(images,(T,X,Y,1))
    pred = self.model.predict(images)
    obs = np.argmax(pred,axis=1)
    return obs

  def loadParameters(self):
    """
    Loads the model and emission matrix
    from the "path_to_parameters" folder
    """
    modelPath = os.path.join(self.path_to_parameters,'model.h5')
    self.model = load_model(modelPath)
    emissionMatrixPath = os.path.join(self.path_to_parameters,'emission_matrix.npy')
    self.emissionMatrix = np.load(emissionMatrixPath)

  def initializeParams(self,case):
    """
    Initializes the transition matrix and the
    initial state probability.

    PARAMS 
    ------
    case : int
      The case, if 1 then allow for transitions 
      back from M to G1 and a general initial 
      probability. If 2 only forward transitions
      allowed and initial probability is 1 for state
      G1

    RETURNS 
    -------
    tM : 4 x 4 numpy array
      The transition matrix
    pi : 4 numpy array
      The initial state probability

    """
    tM = np.zeros((4,4))
    pi = np.zeros(4)
    for i in range(3):
      tM[i,i] = 0.999
      tM[i,i+1] = 0.001
    if case == 1:
      tM[3,3] = 0.999
      tM[3,0] = 0.001
      pi = np.ones(4)/4
    if case == 2:
      tM[3,3] = 1
      pi[0] = 1
    return tM,pi

  def get_state_lengths(self,path):
    """
    Analyses the predicted path

    Takes a path (series of cell cycle stages)
    predicted by the model. Finds the transition 
    times between each state, the lengths of those
    states

    PARAMS
    ------
    path : T numpy array
      The predictions from the model

    RETURNS
    -------
    states : NumTrans numpy array
      The resulting stage after each
      transition.
    lengths : NumTrans numpy array
      The length of each state
    time : NumTrans numpy array
      The time of each transition
      (frame number)

    """
    states = np.array([])
    lengths = np.array([])
    time = np.array([])

    current_state = path[0]
    current_state_length = 0
    last_trans_time = 0
    for t,i in enumerate(path):
      if i == current_state:
        current_state_length += 1
      if i != current_state:
        states = np.append(states,current_state)
        lengths = np.append(lengths,current_state_length)
        time = np.append(time,last_trans_time)
        current_state = i 
        current_state_length = 1
        last_trans_time = t
    states = np.append(states,current_state)
    lengths = np.append(lengths,current_state_length)
    time = np.append(time,last_trans_time)
    return states,lengths,time

  def store_results(self,id_num,path,std):
    """
    Adds results to the dataframe

    PARAMS
    ------
    id_num : int
      The id number of the cel
    path : T numpy array
      The sequence of predicted states
    std : float
      The predicted standard deviation 
      of the posterior distribution of
      states
    """
    num2state = ['G1','S','G2','M']
    st,le,ti = self.get_state_lengths(path)
    for s,l,t in zip(st,le,ti):
      df = pd.DataFrame({'Object No':id_num,'Start Time':int(t),
        'State':num2state[int(s)],'Length':int(l),'Std':std},index=[0])
      self.df = self.df.append(df,ignore_index=True)

  def save_results(self,case,method):
    """
    Saves the data frame with observations as a csv file

    PARAMS
    ------
    case : int
      The case used in predictions
    method : int
      The method used for predictions
    """
    path = os.path.join(self.path_to_save,'M{}C{}.csv'.format(method,case))
    self.df.to_csv(path)
    self.df = pd.DataFrame(columns = ['Object No','Start Time','State','Length','Std'])

  def classify_cell(self,id_num,case=1,method=2):
    """ Make predictions for one cell

    PARAMS
    ------
    id_num : int
      The id number of the cell
    case : int
      The case to use
    method : int
      The method to use
    """
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      path = os.path.join(self.path_to_images,'cell{}.npy'.format(int(id_num)))
      cell = np.load(path)
      cell = cell/np.max(cell)
      tM,pi = self.initializeParams(case)
      obs = self.getObservations(cell)
      self.hmm.update_parameters(obs,pi,tM,self.emissionMatrix)
      self.hmm.fit_parameters(case == 1,method==2,25)
      path,std = self.hmm.predict_path(100)
      self.store_results(id_num,path,std)
      return path,std

  def classify_all_cells(self,case,method,index_file=None):
    """
    Classifies all cells metione in the details.pkl file 
    in the images folder.

    PARAMS
    -----
    case : int
      The case to use
    method : int
      The method to use
    index_file : string
      A file path to a .npy file containing
      the indexes to classify
    """
    path_to_cells = os.path.join(self.path_to_images,'details.pkl')
    df = pd.read_pickle(path_to_cells)
    ids = df['ID']
    case2_ids = df[df['CASE2'] == True]['ID']

    indexes = None
    if index_file == None:
      indexes = ids
    else:
      indexes = np.load(index_file)

    ids = np.array(list(set(ids).intersect(set(indexes))))
    case2_ids = np.array(list(set(case2_ids).intersect(set(indexes))))

    if case == 1:
      print('Classifying Case 1 Using Method {}\n'.format(method))
      num_cells = int(ids.size)
      for i,cell in enumerate(ids):
        progress = (i+1)/num_cells*100 
        sys.stdout.write("\r%d%%" % progress)
        sys.stdout.flush()
        self.classify_cell(cell,1,method)
      self.save_results(1,method)
    if case == 2:
      print('Classifying Case 2 Using Method {}\n'.format(method))
      num_cells = int(case2_ids.size)
      for i,cell in enumerate(case2_ids):
        progress = (i+1)/num_cells*100
        sys.stdout.write("\r%d%%" % progress)
        sys.stdout.flush()
        self.classify_cell(cell,2,method)
      self.save_results(2,method)

if __name__ == '__main__':
  print('########################')
  print('###### Method CNN ######')
  print('########################')
  print('\n')
  print('Enter the path to the folder containing the \n segmented images')
  image_folder = input()
  print('Image folder: {}'.format(image_folder))
  print('\n')
  print('Enter the path to the folder containing the \n model.h5 and emissionMatrix.npy files')
  param_folder = input()
  print('Parameter folder: {}'.format(param_folder))
  print('\n')
  print('Enter the path to the folder in which to \n save the tracks')
  save_folder = input()
  print('Save folder: {}'.format(save_folder))

  pred = cellCyclePredictor(image_folder,param_folder,save_folder)
  pred.classify_all_cells(1,3)
  pred.classify_all_cells(2,3)

  print('Classification Completed')



