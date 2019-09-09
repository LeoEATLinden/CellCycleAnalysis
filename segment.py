import numpy as np 
import pandas as pd 

def convert_to_np(image_path):
  FILE_NAME = "r{}c{}f{}p{}-ch{}sk{}fk1fl1.tiff"
  CHANNELS = 3
  print(sys.argv)
  _,r, c, f, p, sk,folder,saveFolder = sys.argv
  sk = int(sk)

  FILE_PATH = folder + FILE_NAME
  print("Reading Data")

  # Assume the files are in the same folder
  # Get dimension of data!

  im = Image.open(FILE_PATH.format(r,c,f,p,1,1))
  N,M = np.array(im).shape
  print("Dimensions of generated array: ({},{},{},{})".format(sk,CHANNELS,N,M))
  data = np.empty((sk,CHANNELS,N,M))

  for t in range(1,sk+1):
    for ch in range(CHANNELS):
      img_path = FILE_PATH.format(r,c,f,p,ch+1,t)
      im = Image.open(img_path)
      data[t-1,ch,:,:] = np.array(im)

  NEW_NAME = "r{}c{}f{}p{}.npy"
  NEW_PATH = saveFolder + NEW_NAME
  print("Saving Data")
  np.save(NEW_PATH.format(r,c,f,p),data)
  print("Data saved")

def segment_data(image_path,tracking_path,indexes):
  print('Loading Data File')
