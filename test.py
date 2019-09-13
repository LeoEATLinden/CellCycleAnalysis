from methodOne import cellCyclePredictor as pred1
from methodTwo import cellCyclePredictor as pred2

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import os

class Tester:
  """
  This class tests the methods 1 to 3 on a test set.
  It is to be supplied with a folder containing the test
  images. A folder containing the test labels.
  A path to the folder containing all parameters will also
  be expected

  """

  def __init__(self,path_to_images,path_to_labels,path_to_params,path_to_save):
    """
    PARAMETERS
    ----------
    path_to_images : string
      A complete path to the folder containing images
      on the form cellX.npy. The folder is expected to
      contain a file details.pkl storing a dataframe 
      with column ID with the id of all cells to be labeled

    path_to_labels : string
      A complete path to the folder containing the labeled data
      the folder is expected to contain a file indexes.npy that
      contains the cell numbers of all annotated cells.

    path_to_params : string
      A complete path to the folders containg the parameters
      (That is the model.h5, emissionMatrix.npy,mu.npy and sigma.npy)

    path_to_save : string
      A complete path to the folder in which to save 
      the data.

    """
    self.path_to_images = path_to_images
    self.path_to_params = path_to_params
    self.path_to_labels = path_to_labels
    self.path_to_save = path_to_save
    self.p1 = pred1(path_to_images,path_to_params,path_to_save)
    self.p2 = pred2(path_to_images,path_to_params,path_to_save)
    


  def get_reports(self):
    """
    Initializes dataframes for the reports 
    """
    self.errorReport = pd.DataFrame(columns=['Method','Case','Object No',
      'Error','WrongTrans','Precision','Recall','Std'])
    self.stateDist = pd.DataFrame(columns=['Method', 'Case', 'State','Length'])

  def save_reports(self):
    """ 
    Saves the reports 
    """
    self.errorReport.to_csv(os.path.join(self.path_to_save,'errorReport.csv'))
    self.stateDist.to_csv(os.path.join(self.path_to_save,'stateDist.csv'))

  def get_cell_error(self,path,labels,std,method,case,cell_id):
    """
    Computes different error types for each cell and stores
    these in the dataframes

    PARAMS
    ------
    path : numpy array
      The path predicted by the model
    labels : numpy array
      The ground-truth of states
    std : float
      The std of the posterior distribution of paths
    method: int
      The method (1,2 or 3)
    case:int
      The case (1 or 2)
    cell_id : int
      The id number of the cell
    """
    errors = 0
    number_transactions_true = 0
    number_transactions_path = 0

    last_true_state = -1
    last_pred_state = -1

    for state,pred_state in zip(labels,path):

      if state != last_true_state:
        # Actual transition
        number_transactions_true += 1
        last_true_state = state

      if pred_state != last_pred_state:
        # Predicted transition
        number_transactions_path += 1
        last_pred_state = pred_state

      if state != pred_state:
        # Error
        errors += 1

    rep = classification_report(labels,path,output_dict=True)
    precision = rep['macro avg']['precision']
    recall = rep['macro avg']['recall']
    incorr_trans = number_transactions_path - number_transactions_true
    error = errors/path.size
    tempError = pd.DataFrame({'Method':method,'Case':case,'Object No': cell_id,
      'Error':error,'WrongTrans':incorr_trans,'Precision':precision,
      'Recall':recall,'Std':std},index=[0])
    self.errorReport = self.errorReport.append(tempError,ignore_index=True)
    

  def get_state_lengths(self,path,method,case):
    """
    Computes the length of each state in a path and stores this in the
    stateDist dataframe

    PARAMS
    ------
    path : numpy array
      The sequence of states predicted by the model

    method : int
      The model (1,2 or 3)
    case : int
      The case (1 or 2)

    """
    num2state = ['G1','S','G2','M']
    currState = path[0]
    length = 0
    for i,state in enumerate(path):
      if state != currState:
        tempFrame = pd.DataFrame({'Method':method,'Case':case,
          'State':num2state[int(currState)],'Length':length},index=[0])
        self.stateDist = self.stateDist.append(tempFrame,ignore_index = True)
        currState = state
        length = 1
      if state == currState:
        length += 1
    tempFrame = pd.DataFrame({'Method':method,'Case':case,
          'State':num2state[int(currState)],'Length':length},index=[0])
    self.stateDist = self.stateDist.append(tempFrame,ignore_index = True)




  def testMethods(self,method,case):

    """
    Test the method and case supplied

    PARAMS
    ------
    method : int
      The method to test (1,2 or 3)
    case : int
      The case to test (1,2 or 3)
    """

    cell_df_path = os.path.join(self.path_to_images,'details.pkl')
    cell_df = pd.read_pickle(cell_df_path)
    label_df_path = os.path.join(self.path_to_labels,'indexes.npy')
    label_ids = np.load(label_df_path)

    label_path = os.path.join(self.path_to_labels,'labels{}.npy')

    cell_ids = cell_df['ID']
    

    test_set = np.array(list(set(cell_ids).intersection(set(label_ids))))

    for id_num in test_set:
      path = None
      std = None
      label = np.load(label_path.format(int(id_num)))
      if method == 1:
        path,std = self.p1.classify_cell(id_num,case=case)
      if method == 2:
        path,std = self.p2.classify_cell(id_num,case=case)
      self.get_cell_error(path,label,std,method,case,id_num)
      self.get_state_lengths(path,method,case)




