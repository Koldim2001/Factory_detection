# Hardhat_detector
Проект по детектированию людей на объекте с касками и без

### Как запускать код на python 3:
```rb
# Загрузка функции и требуемых библиотек
from predict import *
!pip install -r requirements.txt

# Запуск функции детекции людей:
detect(path='test.jpg')

# Запуск функции детекции касок:
detect(path='test.jpg, hardhat_detection=True)
```
