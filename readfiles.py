
import os
import pandas as pd

class io:

  data_folder = 'data/'
  dirname = None
  project_folder = 'data/projects/'
  tracking_file = None
  folder = None
  cnn_path = None
  emission_matrix_path = None

  project_name = None

  annotations = None

  model = None
  emission_matrix = None

  def __init__(self):
    self.dirname = os.path.dirname(__file__)

    pass

  def get_projects(self):
    path = os.path.join(self.dirname,self.project_folder)
    return os.listdir(path)

  def get_models(self):
    path = os.path.join(self.dirname,self.data_folder)
    path = os.path.join(path,'models/')
    return os.listdir(path)

  def read_indexes():
    pass

  def create_project(self,project_name,image_file,tracking_file):
    self.project_name = project_name
    folder = os.path.join(self.dirname,self.project_folder)
    folder = os.path.join(folder,project_name)
    self.folder = folder
    try:
      os.mkdir(folder)
    except OSError:
      print('Creation of directory {} failed'.format(folder))
    else:
      print('Successfully created the project {}'.format(project_name))
      track = pd.read_csv(tracking_file,sep='\t')
      df = track[['Object No','Timepoint','X','Y']]
      df.columns = ['id','t','x','y']
      tracking_file = os.path.join(folder,'track.csv')
      df.to_csv(tracking_file)
      self.tracking_file = tracking_file

  def load_project(self,project_name):
    self.project_name = project_name
    folder = os.path.join(self.dirname,self.project_folder)
    folder = os.path.join(folder,project_name)
    self.folder = folder

    tracking_file = os.path.join(folder,'track.csv')
    self.tracking_file = tracking_file


  def get_cell_ids(self):
    df = pd.read_csv(tracking_file)
    ids = df['id'].unique()
    length = np.zeros(ids.size,dtype='int')
    for i in ids:
      obj_frame = df[df['id'] == i]
      length[i] = np.max(obj_frame['t'])-np.min(obj_frame['t'])
    return ids,length

  def load_model_paths(model_name):
    data_path = os.path.join(self.dirname,data_folder)
    model_path = os.path.join(data_path,'models/')
    model_path = os.path.join(data_path,model_name)
    cnn_path = os.path.join(model_path,'model.h5')
    emission_path = os.path.join(model_path,'emission_matrix.npy')
    self.model_path = cnn_path
    self.emission_matrix_path = emission_path























