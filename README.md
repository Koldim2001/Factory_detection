# Проект по детекции
Проект по созданию системы безопасности на предприятии, реализующей детектирование людей и автоматическую проверку наличия касок по изображениям с видеокамер

### Как запускать код на python 3:
```rb
pip install -r requirements.txt

python
from predict import *

# Запуск функции детекции людей:
detect(path='test.jpg')

# Запуск функции детекции касок:
detect(path='test.jpg, hardhat_detection=True)
```
