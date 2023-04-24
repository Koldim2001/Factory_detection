
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
    Вычисляет recall для заданного box_iou и порогового значения IoU (IoU threshold).

    Параметры:
    - box_iou: тензор коэффициентов IoU размерности [N, M], где N и M - количество ограничивающих рамок.
    - iou_threshold: пороговое значение IoU, при котором детектированный объект считается правильно классифицированным.

    Возвращает:
    - recall: метрика recall для заданного box_iou и порогового значения IoU.
    """
    import torch

    TP = 0  # количество правильно классифицированных объектов
    FN = 0  # количество неправильно классифицированных объектов

    for i in range(box_iou.shape[1]):
        max_iou = torch.max(box_iou[:, i])  # максимальное значение IoU для i-го истинного объекта
        if max_iou >= iou_threshold:
            TP += 1
        else:
            FN += 1

    recall = TP / (TP + FN)
    return recall


def precision(box_iou, iou_threshold=0.5):
    """
    Вычисляет precision для заданного box_iou и порогового значения IoU (IoU threshold).

    Параметры:
    - box_iou: тензор коэффициентов IoU размерности [N, M], где N и M - количество ограничивающих рамок.
    - iou_threshold: пороговое значение IoU, при котором детектированный объект считается правильно классифицированным.

    Возвращает:
    - precision: точность (precision) для заданного box_iou и порогового значения IoU.
    """
    import torch
    
    TP = 0  # количество правильно классифицированных объектов
    FP = 0  # количество неправильно классифицированных объектов

    for i in range(box_iou.shape[0]):
        max_iou = torch.max(box_iou[i])  # максимальное значение IoU для i-й детектированной рамки
        if max_iou >= iou_threshold:
            TP += 1
        else:
            FP += 1

    precision = TP / (TP + FP)
    return precision


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
           

def calculate_ap(box_iou_list, iou_threshold):
    """
    Вычисляет метрику AP для списка тензоров коэффициентов IoU и порогового значения IoU.

    Параметры:
    - box_iou_list: список тензоров коэффициентов IoU размерности [N, M], где N и M - количество ограничивающих рамок.
    - iou_threshold: пороговое значение IoU, при котором детектированный объект считается правильно классифицированным.

    Возвращает:
    - ap: значение метрики AP для всех изображений в списке box_iou_list.
    """
    import torch 
    from sklearn.metrics import average_precision_score

    y_true = []
    y_scores = []
    for box_iou in box_iou_list:
        y_true += [1 if iou >= iou_threshold else 0 for iou in box_iou]
        y_scores += [iou for iou in box_iou]
    return average_precision_score(y_true, y_scores)
