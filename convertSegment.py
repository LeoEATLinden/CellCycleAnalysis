from PIL import Image
import numpy as np
import pandas as pd
import os 
import sys
import re

class Segment:
  """ This class will segment images

  Attributes:
  ---------
  path_to_images : string
    The path to the folder with the images
  path_to_track1: string
    The path to the tracking file with
    columns Object No, Timepoint, X, Y
  path_to_track2: string
    The path to the file identifying 
    split events. Needs columns
    Object No, Tracks: Tracked Nuclei Selected - Start Type, and
    Tracks: Tracked Nuclei Selected - End Type
  path_to_save: string
    The folder in which to save the segmented images
  saveFrameName : string
    The name of the file containing a pandas data 
    frame with cell id, sequence length and
    weither or not it is a split to split trace.
  FILENAME : string
    The assumed form of the name of the input images
  pcna_ch : int
    The image channel of interest.

  """


  def __init__(self,path_to_images,path_to_track1,path_to_track2,path_to_save,pcna_ch):
    """ Initializes variables
    """
    self.path_to_images = path_to_images
    self.path_to_track1 = path_to_track1
    self.path_to_track2 = path_to_track2
    self.path_to_save = path_to_save
    self.saveFrameName = 'details.pkl'
    self.FILENAME = "r{}c{}f{}p{}-ch{}sk{}fk1fl1.tiff"
    self.pcna_ch = pcna_ch


  def findObjects(self,minLength=0):
    """
    Reads files and ids

    This method reads both tracking files and extract the
    Object No for all tracked cells.

    It also takes the values of r,c,f and p from the first 
    image in the folder.
    """
    selected_objects = np.array([])
    object_file = pd.read_csv(self.path_to_track1,sep='\t')
    track_file = pd.read_csv(self.path_to_track2,sep='\t')
    self.track_file = track_file
    self.object_file = object_file
    object_ids = object_file['Object No'].unique()

    for cell in object_ids:
      cell_length = object_file[object_file['Object No'] == cell]['Timepoint'].size
      if cell_length >= minLength:
        selected_objects = np.append(selected_objects,cell)


    self.object_ids = selected_objects
    image = os.listdir(self.path_to_images)[1]
    nums = re.findall('\d+',image)

    images = os.listdir(self.path_to_images)[1:]
    maxTime = 0
    for image in images:
      time = int(re.findall('\d+',image)[5])
      if time > maxTime:
        maxTime = time

    self.maxTime = maxTime

    self.r = nums[0]
    self.c = nums[1]
    self.f = nums[2]
    self.p = nums[3]


  def extract_subfig(self,x,y, imarray):
    """
    Extracts a subfigure from an image

    Extracts a 32 by 32 pixels subfigure centered around x and y.
    If x and y is too close to any of the edges, the subfigure is 
    translated as to be entierly in the input image.

    Parameters
    ----------
    x : int
      The x coordinate
    y : int 
      The y coordinate
    imarray : 2d numpy array 
      The image to extract from

    Output
    ------
    subfig : a 32 by 32 numpy array
      The extracted image 

    """
  
    N = 16
    ymax,xmax = imarray.shape
    x_bot = x-N
    x_top = x+N
    y_bot = y-N
    y_top = y+N
    if  y_bot< 0:
      y_top = 2*N
      y_bot = 0
    elif y_top > ymax:
      y_top = ymax
      y_bot = ymax-2*N
    if x_bot < 0:
      x_top = 2*N
      x_bot = 0
    elif x_top > xmax:
      x_top = xmax
      x_bot = xmax-2*N
    subfig = imarray[y_bot:y_top,x_bot:x_top]
    return subfig

  def isSplitToSplit(self,id_num):
    """ Checks if a given figure starts and end 
        with a split event

    Parameters
    ----------
    id_num : int
      The Object No of the cell

    Returns
    ------
      boolean to indicate true or false
    """
    startTag = 'Tracks: Tracked Nuclei Selected - Start Type'
    endTag = 'Tracks: Tracked Nuclei Selected - End Type'
    track_file = self.track_file
    begin = (track_file[track_file['Object No'] == id_num][startTag] == 'Split').bool()
    end = (track_file[track_file['Object No'] == id_num][endTag] == 'Split').bool()
    return begin and end

  def extractObjects(self):
    """
    Extract all cells from the images

    For each object id in the tracking file this
    method loads an image for each time point and
    then calls extract_subfig to extract a subfigure 
    with the cell from that image.

    These subfigures are combined into an array following
    the cell for the entire trace

    A dataframe is created to contain the ID, Length 
    and information about if it is a split to split cell.

    """
    cell_file_name = 'cell{}.npy'
    df = pd.DataFrame(columns = ['ID','LENGTH','CASE2'])
    for id_num in self.object_ids:
      id_num = int(id_num)
      cell_trace = self.object_file[self.object_file['Object No'] == id_num]
      X = np.array(cell_trace['X'])
      Y = np.array(cell_trace['Y'])
      # The timepoints in the trace starts with 0 
      # but the image names start with 1
      T = np.array(cell_trace['Timepoint'])+1
      df2 = pd.DataFrame({'ID':id_num,'LENGTH':T.size,'CASE2':self.isSplitToSplit(id_num)},index=[0])
      df = df.append(df2,ignore_index = True)
      cell = np.empty((T.size, 32,32))
      for i,t in enumerate(T):
        path = os.path.join(self.path_to_images,self.FILENAME.format(self.r,self.c,self.f,self.p,self.pcna_ch,t))
        im = np.array(Image.open(path))
        cell[i,:,:] = self.extract_subfig(X[i],Y[i],im)
      save_path = os.path.join(self.path_to_save,cell_file_name.format(id_num))
      np.save(save_path,cell)
    save_path_frame = os.path.join(self.path_to_save,self.saveFrameName)
    df.to_pickle(save_path_frame)

  def extractObjectsParallel(self):
    # Load images
    pathInitial = os.path.join(self.path_to_images,self.FILENAME.format(self.r,self.c,self.f,self.p,self.pcna_ch,1))
    im = np.array(Image.open(pathInitial))
    image_series = np.empty((self.maxTime,im.shape[0],im.shape[1]))

    print('Loading image file:\n')

    for t in range(1,self.maxTime+1):
      progress = t/self.maxTime*100
      sys.stdout.write("\r%d%%" % progress )
      sys.stdout.flush()
      path = os.path.join(self.path_to_images,self.FILENAME.format(self.r,self.c,self.f,self.p,self.pcna_ch,t))
      image_series[t-1] = np.array(Image.open(path))
    cell_file_name = 'cell{}.npy'
    df = pd.DataFrame(columns = ['ID','LENGTH','CASE2'])

    print('\nExtracting cells from images: \n')
    number_of_cells = self.object_ids.size
    for i,id_num in enumerate(self.object_ids):
      progress = (i+1)/number_of_cells*100
      sys.stdout.write("\r%d%%" % progress)
      sys.stdout.flush()
      id_num = int(id_num)
      cell_trace = self.object_file[self.object_file['Object No'] == id_num]
      X = np.array(cell_trace['X'])
      Y = np.array(cell_trace['Y'])
      # The timepoints in the trace starts with 0 
      # but the image names start with 1
      T = np.array(cell_trace['Timepoint'])+1
      df2 = pd.DataFrame({'ID':id_num,'LENGTH':T.size,'CASE2':self.isSplitToSplit(id_num)},index=[0])
      df = df.append(df2,ignore_index = True)
      cell = np.empty((T.size, 32,32))
      for i,t in enumerate(T):
        cell[i,:,:] = self.extract_subfig(X[i],Y[i],image_series[t-1])
      save_path = os.path.join(self.path_to_save,cell_file_name.format(int(id_num)))
      np.save(save_path,cell)
    save_path_frame = os.path.join(self.path_to_save,self.saveFrameName)
    df.to_pickle(save_path_frame)

if __name__ == '__main__':
  
  print('Please enter the path to the folder with the tiff images')
  path_to_images = input()
  print('Please enter the path to the file with cell coordinates')
  path_to_track1 = input()
  print('Please enter the path to the file with start and end events')
  path_to_track2 = input()
  print('Please enter path to folder in which to save')
  path_to_save = input()
  print('Please enter number of pcna_channel')
  ch_num = input()
  ch_num = int(ch_num)

  seg = Segment(path_to_images,path_to_track1,path_to_track2,path_to_save,ch_num)
  seg.findObjects()
  seg.extractObjectsParallel()



