from PIL import Image
import numpy as np
import pandas as pd
import os 
import re

class Segment:
  """ This class will segment microscope images
  """
  def __init__(self,path_to_images,path_to_track,path_to_save,channels,pcna_ch):
    self.path_to_images = path_to_images
    self.path_to_track = path_to_track
    self.path_to_save = path_to_save
    self.channels = channels
    self.FILENAME = "r{}c{}f{}p{}-ch{}sk{}fk1fl1.tiff"
    self.CHIND = 4
    self.pcna_ch = pcna_ch
    self.SKIND = 5

  def findObjects(self):
    object_file = pd.read_csv(self.path_to_track,sep='\t')
    self.object_file = object_file
    object_ids = object_file['Object No'].unique()
    self.object_ids = object_ids
    image = os.listdir(self.path_to_images)[1]
    nums = re.findall('\d+',image)
    self.r = nums[0]
    self.c = nums[1]
    self.f = nums[2]
    self.p = nums[3]

  def extract_subfig(self,x,y, imarray):# Extract this from the image, N pixels in each direction
    N = 16
    # If some part of this extract is out of bounds on the imarray. Translate image
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

  def extractObjects(self):
    cell_file_name = 'cell{}.npy'
    for id_num in self.object_ids[:10]:
      cell_trace = self.object_file[self.object_file['Object No'] == id_num]
      X = np.array(cell_trace['X'])
      Y = np.array(cell_trace['Y'])
      # The timepoints in the trace starts with 0 
      # but the image names start with 1
      T = np.array(cell_trace['Timepoint'])+1
      cell = np.empty((T.size, 32,32))
      for i,t in enumerate(T):
        path = os.path.join(self.path_to_images,self.FILENAME.format(self.r,self.c,self.f,self.p,self.pcna_ch,t))
        im = np.array(Image.open(path))
        cell[i,:,:] = self.extract_subfig(X[i],Y[i],im)
      save_path = os.path.join(self.path_to_save,cell_file_name.format(id_num))
      np.save(save_path,cell)

