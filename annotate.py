import os
import pandas as pd 
import numpy as np 

class Annotate:
  """ This class allows for easy annotation of image series
  """
  def __init__ (self,path_to_images,path_to_save):
    """
    This class allows for easy annotation of image series

    PARAMETERS
    ----------
    path_to_images : string
      A complete path to the folder containing images
      on the form cellX.npy. The folder is expected to
      contain a file details.pkl storing a dataframe 
      with column ID with the id of all cells to be labeled

    path_to_save : string
      A complete path to the folder in which to save 
      the data.

    """
    self.path_to_images = path_to_images
    self.path_to_save = path_to_save
    image_df_path = os.path.join(self.path_to_images,'details.pkl')
    image_df = pd.read_pickle(image_df_path)
    self.image_df = image_df
    self.label_path = os.path.join(self.path_to_save,'indexes.npy')
    annotated = None
    if os.path.isfile(self.label_path):
      annotated = np.load(self.label_path)
    else:
      annotated = np.array([])
    self.annotated = annotated
    self.ids = image_df['ID']

    self.annotate_cells()

  def annotate_cells(self):
    """
    Loops through all cells not yet annotated
    """
    to_annotate = np.array(list(set(self.ids)-set(self.annotated)))
    for cell in to_annotate:
      self.annotate_cell(cell)

  def annotate_cell(self,cell):
    """
    Annotate one cell

    Loads the cell image to determine the length, N, of the sequence
    Then asks for the initial cell cycle stage. This can be given 
    as either G1,S,G2 and M or as 0,1,2,3.

    Then enter the time point for next cell cycle transition.
    This continues untill either the time is set ot be greater than N
    or q is entered.
    """
    img = np.load(os.path.join(self.path_to_images,'cell{}.npy'.format(cell)))
    N = img.shape[0]
    label = np.zeros(N)
    print('Enter first state of cell {}'.format(cell))
    state = input()
    state2num = {'G1':0,'0':0,'S':1,'1':1,'G2':2,'2':2,'M':3,'3':3}
    state = state2num[state]
    p = 0
    while True:
      print('Enter time of next transition: (if finished enter q')
      t = input()
      if t == 'q':
        label[p:] = state
        break
      t = int(t)
      if t > N:
        label[p:] = state 
        break
      label[p:t] = state
      p = t
      state = (state+1)%4
    self.annotated = np.append(self.annotated,cell)
    np.save(self.label_path,self.annotated)
    np.save(os.path.join(self.path_to_save,'label{}.npy'.format(cell)), label)
