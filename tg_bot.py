
import cv2
import numpy as np
import os
from telegram import Update
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import warnings
warnings.filterwarnings("ignore")


# Функция для обработки сообщений с фотографией
def process_photo(update: Update, context: CallbackContext) -> None:
    # Получаем фотографию из сообщения
    photo_file = update.message.photo[-1].get_file()
    # Генерируем имя файла для сохранения
    # Сохраняем фотографию
    photo_file.download('file_tg.jpg')

    ## ML часть:
    classes = ['-','person']
    num_classes_detect = len(classes)

    # Загрузка модели Faster R-CNN и ее весов
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Заменим число выходных класссов на то, что нам нужно
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_detect)
    model.load_state_dict(torch.load('models/model_human_detection_final.pth'))

    # Загрузка изображения и его преобразование в тензор
    image = cv2.imread('file_tg.jpg')
    image_tensor = torchvision.transforms.functional.to_tensor(image)

    # Получение предсказаний на основе загруженной модели и тензора с изображением
    model.eval()
    with torch.no_grad():
        predictions = model([image_tensor])

    # Визуализация полученных bounding boxes с подписью класса детекции
    for i, prediction in enumerate(predictions[0]['boxes']):
        if predictions[0]['scores'][i] < 0.85:
            continue
        x1, y1, x2, y2 = prediction.tolist()
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
    dict_classes = {} 
    for i, prediction in enumerate(predictions[0]['labels']):
        if predictions[0]['scores'][i] < 0.85:
            continue
        label_code = prediction.item()
        label_name = classes[label_code]

        # Словарь в котором будут указано сколько объектов каждого класса обнаружено:
        if label_name not in dict_classes:
            dict_classes[label_name] = 0
        dict_classes[label_name] += 1

        x1, y1, x2, y2 = predictions[0]['boxes'][i].tolist()
        cv2.putText(image, label_name, (int(x1), int(y1)-5), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Вывод числа задетектированных объектов:
    for key in dict_classes:
        print(f"Объектов класса {key} обнаружено {dict_classes[key]}")
        text_output = 'Людей обнаружил: '+ str(dict_classes[key])
    if dict_classes == {}:
        print(f"Ни один объект не обнаружен")
        text_output = 'Ни одного человека не обнаружил'
        context.bot.send_message(chat_id=update.message.chat_id, text=text_output)
        return

    isWritten = cv2.imwrite('temp_res.jpg', image)
    if isWritten:
        print('Image is successfully saved as file.')
    else:
        print('problem')    

    ## БОТ:
    context.bot.send_message(chat_id=update.message.chat_id, text=text_output)

    # Загружаем фотографию в cv2
    img = cv2.imread('temp_res.jpg')
    # Преобразуем в чб версию
    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Создаём объект байтов для отправки
    ret, img_byte = cv2.imencode('.jpg', img)
    img_bytes = img_byte.tobytes()
    # Отправляем чб версию фотографии в чат
    context.bot.send_photo(chat_id=update.message.chat_id, photo=img_bytes)

# Инициализация бота
updater = Updater("ВСТАВИТЬ НАДО СЮДА API ключ от своего бота")

# Добавляем обработчик сообщений с фотографией
updater.dispatcher.add_handler(MessageHandler(Filters.photo, process_photo))

# Запускаем бота
updater.start_polling()
updater.idle()
