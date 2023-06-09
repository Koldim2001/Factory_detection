
def calculate_iou(model, dataset, treshold=0.85):
    '''
    Функция calculate_iou прининимает на вход модель, датасет и порог score. 
    На выходе выдает массив с числом элементов равным чилу объектов в датасете 
    и содержащим массив соответвий IOU между предсказанным и рельным bounding боксом
    '''
    from torchvision.ops.boxes import box_iou
    import torch
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    iou_scores = []
    for i in range(len(dataset)):
        img, target = dataset[i]
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(img)
        predicted_boxes = prediction[0]['boxes'].cpu()
        predicted_boxes_real = []

        # Выберем предикты с уверенность выше порога:
        for i, box in enumerate(predicted_boxes):
            if (prediction[0]['scores'][i] > treshold):
                predicted_boxes_real.append(box.tolist())
        predicted_boxes_real = torch.as_tensor(predicted_boxes_real)
        
        target_boxes = target['boxes']
        iou = box_iou(predicted_boxes_real, target_boxes)
        iou_scores.append(iou)
    return iou_scores


def recall(box_iou, iou_threshold=0.5):
    """
    Рассчитывает метрику precision для заданного тензора коэффициентов IoU и порогового значения IoU.

    Параметры:
    - box_iou: тензор коэффициентов IoU размерности [N, M], где N и M - количество ограничивающих рамок.
    - iou_threshold: пороговое значение IoU, при котором детектированный объект считается правильно классифицированным.

    Возвращает:
    - recall: метрика recall для заданного тензора коэффициентов IoU и порогового значения IoU.
    """
    import torch 

    num_real_boxes = box_iou.shape[1]  # количество реальных объектов

    num_correct_boxes = 0  # количество правильно классифицированных объектов

    for i in range(box_iou.shape[1]):
        max_iou = torch.max(box_iou[:, i])  # максимальное значение IoU для i-го обнаруженного объекта
        if max_iou >= iou_threshold:
            num_correct_boxes += 1
        
    recall = num_correct_boxes / num_real_boxes if num_real_boxes > 0 else 0.0
    return recall


def precision(box_iou, iou_threshold=0.5):
    """
    Рассчитывает метрику precision для заданного тензора коэффициентов IoU и порогового значения IoU.

    Параметры:
    - box_iou: тензор коэффициентов IoU размерности [N, M], где N и M - количество ограничивающих рамок.
    - iou_threshold: пороговое значение IoU, при котором детектированный объект считается правильно классифицированным.

    Возвращает:
    - precision: метрика precision для заданного тензора коэффициентов IoU и порогового значения IoU.
    """
    import torch
    
    num_correct_boxes = 0  # количество правильно классифицированных объектов

    for i in range(box_iou.shape[0]):
        max_iou = torch.max(box_iou[i, :])  # максимальное значение IoU для i-го обнаруженного объекта
        if max_iou >= iou_threshold:
            num_correct_boxes += 1

    delta = num_correct_boxes - box_iou.shape[1]
    if delta <= 0:
        return 1
    else:
        return box_iou.shape[1] / num_correct_boxes


def mean_metric(list_iou_boxes, func, iou_treshold=0.5):
    ''' 
    Считает среднее значение метрики по датасету
    '''
    metric = []
    for i in list_iou_boxes:
        if func == 'recall':
            metric.append(recall(box_iou=i, iou_threshold=iou_treshold))
        if func == 'precision':
            metric.append(precision(box_iou=i, iou_threshold=iou_treshold))
    return sum(metric) / len(list_iou_boxes)
        

def average_precision(recall, precision):
    """
    Функция для подсчета average precision.

    Аргументы:
    recall -- список значений recall
    precision -- список значений precision

    Возвращает:
    ap -- значение average precision
    """

    # Добавляем 0 в начало и конец списка recall, чтобы
    # перейти к правильной кривой precision-recall.
    recall = [0.0] + recall + [1.0]

    # Добавляем 0 в начало и конец списка precision, чтобы
    # перейти к правильной кривой precision-recall.
    precision = [0.0] + precision + [0.0]

    # Считаем площадь под кривой precision-recall.
    for i in range(len(precision)-2, -1, -1):
        precision[i] = max(precision[i], precision[i+1])
    ap = 0.0
    for i in range(1, len(recall)):
        if recall[i] != recall[i-1]:
            ap += ((recall[i] - recall[i-1]) * precision[i])

    # Возвращаем значение average precision.
    return ap


def mAP_AP_dataset(dataset, model, multiclasses=False):
    '''
    Функция по подсчету mAP и AP по всему валидационному датасету. 
    Исользует готовый покет torchmetrics.detection для релизации
    multiclasses=False => считать AP
    multiclasses=True => считать mAP
    '''
    from tqdm import tqdm
    import torch
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    device = "cpu"
    model.to(device)
    predict_list=[]
    target_list=[]

    # Пройдем по всему датасету чтобы сложить результаты предиктов и truth в один суммарный список
    model.eval()
    for i in tqdm(range(len(dataset))):
        img, target = dataset[i]
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(img)
        predict_list.append(prediction[0])
        target_list.append(target)

    if multiclasses:
        metric = MeanAveragePrecision(class_metrics=True)
        metric.update(predict_list, target_list)
        metrics = metric.compute()

        # Вывод результатов в консоль:
        print('\n')
        print('Значения Average Precision для каждого класса:')
        print('AP (среднее по порогам IoU=.50:.05:.95) для класса WITH HARDHAT =',
            round(float(metrics['map_per_class'][0]),4))
        print('AP (среднее по порогам IoU=.50:.05:.95) для класса WITHOUT HARDHAT =',
            round(float(metrics['map_per_class'][1]),4))
        print('\n')
        print('Значения Mean Average Precision:')
        print('mAP (среднее по порогам IoU=.50:.05:.95) =',
            round(float(metrics['map']),4))
        print('mAP (при IoU=.50) =', round(float(metrics['map_50']), 4))
        print('mAP (при IoU=.70) =', round(float(metrics['map_75']), 4))
        print('mAP (для малых объектов, у которых площадь < 32*32 пикселя) =',
            round(float(metrics['map_small']),4))
        print('mAP (для средних объектов, у которых площадь от 32*32 до 64*64 пикселя) =',
            round(float(metrics['map_medium']),4))
        print('mAP (для больших объектов, у которых площадь > 64*64 пикселя) =',
            round(float(metrics['map_large']),4))
    else:
        print('\n')
        metric = MeanAveragePrecision()
        metric.update(predict_list, target_list)
        metrics = metric.compute()
        print('Значения Average Precision для класса person:')
        print('AP (при IoU=.50) =', round(float(metrics['map_50']), 4))

        


        