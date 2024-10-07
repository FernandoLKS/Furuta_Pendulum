import os
import pandas as pd

class FileManipulation:
    def __init__(self, folder_name='output'):
        self.folder_name = folder_name
        self.create_folder()

    def create_folder(self):
        try:
            if not os.path.exists(self.folder_name):
                os.makedirs(self.folder_name)
                print(f"Folder '{self.folder_name}' created successfully.")
            else:
                print(f"Folder '{self.folder_name}' already exists.")
        except Exception as e:
            print(f"An error occurred while creating the folder: {e}")

    def save_data_frame(self, df, file_name):
        try:
            file_path = os.path.join(self.folder_name, file_name)
            df.to_csv(file_path, index=True)
            print(f"DataFrame saved successfully at '{file_path}'!")
        except Exception as e:
            print(f"An error occurred while saving the DataFrame: {e}")

    def save_graphic(self, graphic, file_name):
        try:
            file_path = os.path.join(self.folder_name, file_name)
            graphic.savefig(file_path)
            print(f"Graphic saved successfully at '{file_path}'!")
        except Exception as e:
            print(f"An error occurred while saving the graphic: {e}")

    def read_data_frame(self, file_name):
        try:
            file_path = os.path.join(self.folder_name, file_name)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0)
                print(f"DataFrame loaded successfully from '{file_path}'!")
                return df
            else:
                print(f"The file '{file_path}' does not exist.")
                return None
        except Exception as e:
            print(f"An error occurred while reading the DataFrame: {e}")
            return None
