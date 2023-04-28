
def detect_and_visualize(image_input,
                         treshhold=0.85,
                         classes=['-', 'person'],
                         model_path='models/model_human_detection.pth',
                         plt_show='False',
                         reshape=False):
    '''
    Функция detect_and_visualize реализует загрузку нужной модели и построение
    итогового изображения, на котором расставлены bounding боксы и подписаны классы
    На вход подается image_input в виде пути к файлу 
    treshhold - значение минимального порога уверенности классификатора в боксе
    classes - list, содержащий наименования классов
    plt_show = 'False'-тогда выводим ответ в отельное окно.
    Если True, то выводим в jupiter notebook
    reshape = True переводит изображение в размер 720 на 480.
    По умолчанию False (исходный размер)

    Так же данная функция выдает на выходе число задетектированных объектов каждого класса
    '''
    import torch
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    import cv2
    import warnings
    from matplotlib import pyplot as plt
    warnings.filterwarnings("ignore")
    
    if classes[0] != '-':
        classes = ['-'] + classes
    
    num_classes_detect = len(classes)
    # Загрузка модели Faster R-CNN и ее весов
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Заменим число выходных класссов на то, что нам нужно
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_detect)
    model.load_state_dict(torch.load(model_path))

    # Загрузка изображения и его преобразование в тензор
    image = cv2.imread(image_input)
    image_tensor = torchvision.transforms.functional.to_tensor(image)

    # Получение предсказаний на основе загруженной модели и тензора с изображением
    model.eval()
    with torch.no_grad():
        predictions = model([image_tensor])

    # Визуализация полученных bounding boxes с подписью класса детекции
    for i, prediction in enumerate(predictions[0]['boxes']):
        if predictions[0]['scores'][i] < treshhold:
            continue
        x1, y1, x2, y2 = prediction.tolist()
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
    dict_classes = {} 
    for i, prediction in enumerate(predictions[0]['labels']):
        if predictions[0]['scores'][i] < treshhold:
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
    if dict_classes == {}:
        print(f"Ни один объект не обнаружен")
    
    
    # Изменим размер при необходимости:
    if reshape:
        image = cv2.resize(image, (720, 480))
    
    # Демонстрация результатов:
    if plt_show:
        plt.figure(figsize=(9, 7), dpi=80)
        plt.title(f'Prediction')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.show()
    else:
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def visualize_detection(dataset, model, idx):
    '''
    Данная функция визуализирует для заданной фотографии из датасета рельный
    и предсказанный bounding боксы. Над боксом указывает значение IOU
    '''
    import cv2
    from torchvision.ops.boxes import box_iou
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torchvision.transforms.functional as TF

    # Получаем изображение и цель для заданного индекса
    image, target = dataset[idx]
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Переводим модель в evaluation mode
    model.eval()

    # Получаем предсказания
    with torch.no_grad():
        prediction = model([image.to(device)])
    
    # Получаем предсказанные ограничивающие рамки и соответствующие метки
    predicted_boxes = prediction[0]['boxes'].cpu()
    predicted_labels = prediction[0]['labels'].cpu()
    
    # Получаем настоящие ограничивающие рамки и соответствующие метки
    true_boxes = target['boxes']
    true_labels = target['labels']
    
    # Преобразуем тензор изображения PyTorch в массив NumPy
    image = TF.to_pil_image(image)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Рисуем предсказанные рамки зеленым цветом
    for box in predicted_boxes:
        x1, y1, x2, y2 = box.int()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 225, 50), 3)
        
    # Реальные боксы рисуем красными
    for box in true_boxes:
        x1, y1, x2, y2 = box.int()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 20, 255), 3)
    
    # Вычисляем IOU для каждой пары боксов predict & truth
    ious = []
    for i, predicted_box in enumerate(predicted_boxes):
        iou_max = 0
        for j, true_box in enumerate(true_boxes):
            
            iou = box_iou(torch.tensor([predicted_box.tolist()]),
                          torch.tensor([true_box.tolist()]))
            if iou > iou_max:
                iou_max = iou
        ious.append(iou_max)
    
    # Добавляем текст сверху со значением IOU
    for i, iou in enumerate(ious):
        x1, y1, _, _ = predicted_boxes[i].int()
        cv2.putText(image, f'{float(iou[0][0]):.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)
    
    plt.figure(figsize=(10, 5))
    plt.title('IOU')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    

def download_models(folder_name='models'):
    '''
    Функция для загрузки моделей обученных нейронных сетей в папку folder_name 
    из моего гугл диска
    '''
    import gdown
    import os

    # Создание пустой папки 
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    print('Загружаю модели нейронных сетей:')
    url_1 = "https://drive.google.com/uc?id=15BK4n4NjAahbv6yImTJgLMCsQAeX1LCC"
    output_1 = folder_name + "/model_human_detection_final.pth"
    url_2 = "https://drive.google.com/uc?id=1_66F4THF8-Nws3yodKWxSdXjrmoFANBF"
    output_2 = folder_name + "/model_hardhat_detection_final.pth"

    gdown.download(url_1, output_1, quiet=False)
    gdown.download(url_2, output_2, quiet=False)


def detect(path, hardhat_detection=False):
    '''
    Главная функция проекта. Реализует загрузку моделей, осуществяет детекцию любого входного изображения.

    На вход функции подаются: \n
      path - путь к изображению \n
      hardhat_detection (по умолчанию False) - выбирает вид детекции.
      Если False, то осуществляется детекция людей (1 класс).
      Если True, то определяются 2 класса: человек к каской или человек без каски
    '''
    import os

    # Загрузка с гугл диска обученных моделей
    if not (os.path.exists('models/model_hardhat_detection_final.pth') and 
            os.path.exists('models/model_human_detection_final.pth')):
        download_models(folder_name='models')

    if hardhat_detection:
        detect_and_visualize(path,
                             treshhold=0.5,
                             classes=['hardhat','no_harhat'],
                             model_path='models/model_hardhat_detection_final.pth')
    else:
        detect_and_visualize(path,
                             treshhold=0.85,
                             classes=['person'],
                             model_path='models/model_human_detection_final.pth')
    
    

        
