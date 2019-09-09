
import ui
import readfiles


def manage_project(rw):
  while True:
    choice = print(menue(rw.project_name))
    if choice == 1:
      models = rw.get_models()
      model_name = ui.choose_model(models)
      rw.load_model_paths(model_name)

    if choice == 2:
      index,length = rw.get_cell_ids()
      ids = ui.select_indexes(index,length)
      
      pass

    if choice == 3:
      # Analyse cell traces
      pass

    if choice == 'q':
      # Quit program
      pass

def main():

  rw = readfiles.io()
  ui.print_welcome()
  folders = rw.get_projects()
  choice = ui.get_startup_choice(folders)
  if choice == 'new':
    # Create new project
    name,image_path,tracking_path = ui.create_project()
    rw.create_project(name,image_path,tracking_path)
  else: 
    rw.load_project(name)




if __name__ == '__main__':
  main()
