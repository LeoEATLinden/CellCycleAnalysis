import os
from gaussianHMM import gaussianHMM
import pandas as pd
import sys
import numpy as np
import warnings

class cellCyclePredictor:
  """ cellCyclePredictor

      This class implements the cellCyclePrediction
      method 1. In short it reads the images 
      from a "cellX.npy" file, calculates the mean 
      intensity of each image as well as the std of 
      each image.

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
        A instance of the gaussianHMM class
      df : pandas DataFrame
        the dataframe to save the predictions in
      mu : numpy array
        The mean values for the gaussian in each state
      sigma : numpy arrray
        The covariance matrix for the gaussian in each state


  """


  def __init__(self,path_to_images,path_to_parameters,path_to_save):
    self.path_to_images = path_to_images
    self.path_to_parameters = path_to_parameters
    self.path_to_save = path_to_save
    self.image_info_file = os.path.join(path_to_images,'details.pkl')
    self.hmm = gaussianHMM()
    self.df = pd.DataFrame(columns = ['Object No','Start Time','State','Length','Std'])
    self.loadParameters()

  def getObservations(self,image):
    """
    Takes an image file and returns 
    its mean and standard deviation

    PARAMS
    ------
    images : Tx32 x 32 numpy array
      The input images

    RETURNS
    -------
    obs : Tx2 numpy array
      The mean and std
    """
    T = image.shape[0]
    obs = np.zeros((T,2))
    for t in range(T):
      obs[t,0] = np.mean(image[t,:,:])
      obs[t,1] = np.std(image[t,:,:])
    return obs

  def loadParameters(self):
    """
    Loads the mean vector and the
    covariance matrixes.
    """

    muPath = os.path.join(self.path_to_parameters,'mu.npy')
    sigmaPath = os.path.join(self.path_to_parameters,'sigma.npy')
    self.mu = np.load(muPath)
    self.sigma = np.load(sigmaPath)

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

  def save_results(self,case):
    """
    Saves the data frame with observations as a csv file

    PARAMS
    ------
    case : int
      The case used in predictions
    method : int
      The method used for predictions
    """

    path = os.path.join(self.path_to_save,'M1C{}.csv'.format(case))
    self.df.to_csv(path)
    self.df = pd.DataFrame(columns = ['Object No','Start Time','State','Length','Std'])

  def classify_cell(self,id_num,case=1):
    """ Make predictions for one cell

    PARAMS
    ------
    id_num : int
      The id number of the cell
    case : int
      The case to use

    """
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      path = os.path.join(self.path_to_images,'cell{}.npy'.format(int(id_num)))
      cell = np.load(path)
      obs = self.getObservations(cell)
      tM,pi = self.initializeParams(case)
      self.hmm.update_parameters(obs,pi,tM,self.mu,self.sigma)
      self.hmm.fit_parameters(case == 1,25)
      path,std = self.hmm.predict_path(100)
      self.store_results(id_num,path,std)

  def classify_all_cells(self,case):
    """
    Classifies all cells metione in the details.pkl file 
    in the images folder.

    PARAMS
    -----
    case : int
      The case to use

    """

    path_to_cells = os.path.join(self.path_to_images,'details.pkl')
    df = pd.read_pickle(path_to_cells)
    ids = df['ID']
    case2_ids = df[df['CASE2'] == True]['ID']
    if case == 1:
      print('Classifying Case 1 \n')
      num_cells = int(ids.size)
      for i,cell in enumerate(ids):
        progress = (i+1)/num_cells*100 
        sys.stdout.write("\r%d%%" % progress)
        sys.stdout.flush()
        self.classify_cell(cell,1)
      self.save_results(1)
    if case == 2:
      print('Classifying Case 2 \n')
      num_cells = int(case2_ids.size)
      for i,cell in enumerate(case2_ids):
        progress = (i+1)/num_cells*100
        sys.stdout.write("\r%d%%" % progress )
        sys.stdout.flush()
        self.classify_cell(cell,2)
      self.save_results(2)

if __name__ == '__main__':
  print('########################')
  print('####### Method 1 #######')
  print('########################')
  print('\n')
  print('Enter the path to the folder containing the \n segmented images')
  image_folder = input()
  print('Image folder: {}'.format(image_folder))
  print('\n')
  print('Enter the path to the folder containing the \n mu.npy and sigma.npy files')
  param_folder = input()
  print('Parameter folder: {}'.format(param_folder))
  print('\n')
  print('Enter the path to the folder in which to \n save the tracks')
  save_folder = input()
  print('Save folder: {}'.format(save_folder))

  pred = cellCyclePredictor(image_folder,param_folder,save_folder)
  pred.classify_all_cells(1)
  pred.classify_all_cells(2)

  print('Classification Completed')



