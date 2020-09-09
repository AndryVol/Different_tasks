import sys
import torch
import pandas as pd
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import os
import json

if __name__ == "__main__":
    path_img = sys.argv[1] # записываем путь к папке с изображениями
    path_folder = os.path.dirname(os.path.abspath(sys.argv[0])) # определяем путь к папке с скриптом
    
class CustomDataSet(torch.utils.data.Dataset): # определяем класс CustomDataSet который на вход принимает путь к файлу и то как надо преобразовывать изображение
    def __init__(self, main_dir, transform):   # на выходе тензор изображений
        self.main_dir = main_dir
        self.transform = transform
        self.total_imgs = os.listdir(main_dir) # получаем список всех файлов в папке

    def __len__(self):
        return len(self.total_imgs) # определяем количество файлов

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx]) # создаем путь к одному изображению
        image = Image.open(img_loc).convert('RGB') # загружаем его
        tensor_image = self.transform(image) # преобразуем его
        return tensor_image

class LeNet5(torch.nn.Module): # определяем класс модели, аналоигчно как в determine_gender.ipynb
    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.conv1 = torch.nn.Conv2d( 
            in_channels=3, out_channels=6, kernel_size=5, padding=2) # определение сверточного слоя, с 3 входными каналами, 6 сверток, размер свертки 5х5, паддинг 2
        self.act1  = torch.nn.Tanh() # определение функции активации
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2) # определение пулинг слоя с ядром 2х2, шагом 2
       
        self.conv2 = torch.nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, padding=0) # определение сверточного слоя, с 6 входными каналами, 16 сверток, размер свертки 5х5, паддинг 0
        self.act2  = torch.nn.Tanh() # определение функции активации
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2) # определение пулинг слоя с ядром 2х2, шагом 2

        self.conv3 = torch.nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=2, padding=0) # определение сверточного слоя, с 16 входными каналами, 16 сверток, размер свертки 2х2, паддинг 0
        self.act3  = torch.nn.Tanh() # определение функции активации
        self.pool3 = torch.nn.AvgPool2d(kernel_size=2, stride=1) # определение пулинг слоя с ядром 2х2, шагом 1     
        
        self.fc1   = torch.nn.Linear(2304, 120) # определение линейного слоя с 2304 входами и 120 выходами
        self.act4  = torch.nn.Tanh() # определение функции активации
        
        self.fc2   = torch.nn.Linear(120, 84) # определение линейного слоя с 120 входами и 84 выходами
        self.act5  = torch.nn.Tanh() # определение функции активации
        
        self.fc3   = torch.nn.Linear(84, 2) # определение линейного слоя с 84 входами и 2 выходами

    
    def forward(self, x): # построение нейронной сети
        
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.act3(x)
        x = self.pool3(x)       
        
        x = x.view(x.size(0), -1) # преобразование 4-х мерного массива [batch, n_layer, w, h] в двумерный
 
        x = self.fc1(x)
        x = self.act4(x)
        x = self.fc2(x)
        x = self.act5(x)
        x = self.fc3(x)

        return x
    
def predict_folder_img(img_folder_path): # определяем функцию для создания списка результатов определения пола
    result = pd.DataFrame(columns = ['name', 'prediction']) # создаем таблицу, в которую буем записвать имена файлов и результаты определения пола
    classes = ['female', 'male'] 
    transform = transforms.Compose([transforms.RandomResizedCrop(64), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    my_dataset = CustomDataSet(img_folder_path, transform=transform) # делаем датасет из файлов
    data_loader = torch.utils.data.DataLoader(my_dataset, shuffle=False, batch_size=1) # загружаем данные
    
    model = torch.load(os.path.join(path_folder, 'model')) # загружаем модель
    
    prediction = []
    for i, data in enumerate(data_loader, 0): # оперяделяем пол в каждом файле и записываем результат в prediction
      preds = model.forward(data).argmax(dim=1)
      prediction.append(classes[preds])
      
    result['name'] = os.listdir(img_folder_path) # записываем в таблицу имена файлов
    result['prediction'] = prediction # записываем результаты модели
    result = result.set_index('name')
    dict_img = result['prediction'].to_dict() # преобразуем результаты в словарь
    return json.dumps(dict_img) 

result = predict_folder_img(path_img)

with open(os.path.join(path_folder, 'process_results.json'), 'w') as fw:
  fw.write(result) # записываем результат в файл process_results.json

