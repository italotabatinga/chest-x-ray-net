from pandas import read_csv, DataFrame
from os import path
import numpy as np

#images folder path
images_path = path.join('chexpert', 'train.csv')
diseases = ['No Finding',	'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',	'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',	'Pleural Effusion', 'Pleural Other',	'Fracture',	'Support Devices']

def data_entries(data_path:str=images_path) -> DataFrame:
  return read_csv(data_path)

def get_diseases() -> list:
  diseases_enum = {}

  for idx, disease in enumerate(diseases):
    diseases_enum[disease] = idx
    diseases_enum[idx] = disease

  return diseases_enum

def _get_label_array(diseases_str, diseases_enum, num_diseases):
  label_array = np.zeros(num_diseases)

  indices =[diseases_enum[disease] for disease in diseases_str.split('|')]
  label_array[indices] = 1

  return label_array

def get_label_column(df: DataFrame):
  # diseases = get_diseases()
  # row_count = len(df)
  # diseases_column = df.get('Finding Labels')

  
  # label_column = np.zeros((row_count, len(diseases)//2))
  return df[diseases].applymap(lambda x: x if x == 1 else 0)
  # for index, row in enumerate(label_column):
    
  #   indices = [diseases[disease] for disease in diseases_column[index].split('|')]
  #   row[indices] = 1

  # return label_column

def get_data_column(df: DataFrame):
  return df.get('Path').array

def load_data(df: DataFrame):
  return (get_data_column(df), get_label_column(df))

if __name__ == '__main__':
  data = data_entries()

  # get_label_column(data)
  # print(get_label_column(data).shape)
  # print(get_diseases(data))
  # print(list(data.columns))
  # print(data_entries().columns[0:1]