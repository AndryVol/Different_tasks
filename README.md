Описание файлов:
- max_sum_array.ipynb - Функция для нахождения в массиве непрерывного подмассива с максимальной суммой.
- determine_gender.ipynb - Ноутбук с решением задачи по определению пола человека по фотографии, с использованием нейросети.
- model - Модель обученной нейросети. 
- detrmine_folder.py - Скрипт, который через консоль на вход принимает адрес папки с изображениями, а на выходе дает json файл с результатами для каждого изображения.

### Описание работы detrmine_folder.py:  
Через командную строку вводится команда:  
python <адрес скрипта> <адрес папки с изображениями>  
В результате работы скрипта создается файл *process_results.json* с результатами в виде {"000001.jpg": "female", "000004.jpg": "female"...
Для работы необходимо:  
модель обученной нейросети *model* положить в одну папку с *detrmine_folder.py*
