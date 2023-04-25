'''
Функция detect_and_visualize реализует загрузку нужной модели и построение
итогового изображения, на котором расставлены bounding боксы и подписаны классы
На вход подается image_input в виде пути к файлу или уже трехмерной матрицы [h,w,c]
treshhold - значение минимального порога уверенности классификатора в боксе
classes - list, содержащий наименования классов
plt_show = 'False'-тогда выводим ответ в отельное окно. Если True, то выводим в jupiter notebook
reshape = True переводит изображение в размер 720 на 480. По умолчанию False (исходный размер)

Так же данная функция выдает на выходе число задетектированных объектов каждого класса
'''


def detect_and_visualize(image_input,
                         treshhold=0.85,
                         classes=['-', 'person'],
                         model_path='models/model_human_detection.pth',
                         plt_show='False',
                         reshape=False):
    import torch
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    import cv2
    import warnings
    warnings.filterwarnings("ignore")

    if plt_show:
        from  matplotlib import pyplot as plt

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
    if type(image_input) == str:
        image = cv2.imread(image_input)
    else:
        image = image_input
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
        dict_classes[label_name] +=1

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

    import cv2
    from torchvision.ops.boxes import box_iou
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torchvision.transforms.functional as TF

    # Get the image and the target for the given index
    image, target = dataset[idx]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Put the model in evaluation mode
    model.eval()

    # Make predictions
    with torch.no_grad():
        prediction = model([image.to(device)])
    
    # Get the predicted bounding boxes and their corresponding labels
    predicted_boxes = prediction[0]['boxes'].cpu()
    predicted_labels = prediction[0]['labels'].cpu()
    
    # Get the ground truth bounding boxes and their corresponding labels
    true_boxes = target['boxes']
    true_labels = target['labels']
    
    # Convert the PyTorch tensor image to a NumPy array
    image = TF.to_pil_image(image)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Draw the predicted boxes in red
    for box in predicted_boxes:
        x1, y1, x2, y2 = box.int()
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
    # Draw the ground truth boxes in green
    for box in true_boxes:
        x1, y1, x2, y2 = box.int()
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Calculate the IOU for each predicted box and the corresponding true box
    ious = []
    for i, predicted_box in enumerate(predicted_boxes):
        iou_max = 0
        for j, true_box in enumerate(true_boxes):
            iou = box_iou(predicted_box, true_box)
            if iou > iou_max:
                iou_max = iou
        ious.append(iou_max)
    
    # Add the IOU values as text to the image
    for i, iou in enumerate(ious):
        x1, y1, _, _ = predicted_boxes[i].int()
        cv2.putText(image, f'{iou:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Show the image using matplotlib
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
