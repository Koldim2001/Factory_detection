
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
    for i in tqdm(range(len(dataset))):
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
        

def precision_recall_curve(model, dataset, treshold_score=0.85, n_steps=100):
    '''
    Создает списки list_recall, list_precission для значений метрик точность и полнота
    при значениях порога IOU от 0 до 1 шагом n_steps
    treshold_score - входной параметр минимального скора для оценуи результата детекции
    '''
    import numpy as np

    iou_scores_list = calculate_iou(model, dataset, treshold=treshold_score)
    tresh = np.array(range(n_steps + 1)) / 10
    list_precission = []
    list_recall = []
    for val in tresh:
        list_precission.append(mean_metric(iou_scores_list,
                                            func='precision',
                                            iou_treshold=val))
        list_recall.append(mean_metric(iou_scores_list,
                                    func='recall',
                                    iou_treshold=val))
    return list_recall, list_precission


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