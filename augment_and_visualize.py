''' 
Запуск данной функции создает папку, в которой
находятся аугментированные изображения и аннотации, а также 
исходные варианты этих изображений и аннотаций.
Благодаря отображению исходных файлов вокруг вертикали 
данная функция увеличивает объем исходного датасета в 2 раза

Параметры функции:
  image_dir - Путь до папки с фотографиями
  xml_dir - Путь до папки с xml-файлами
  out_folder - Итоговая папка с аугментироанными данными
'''


def aug(image_dir="detect_dataset/images",
                  xml_dir="detect_dataset/annotations/PASCAL_VOC_xml",
                  out_folder='augmented_dataset'):
    import os
    from PIL import Image
    import xml.etree.ElementTree as ET

    # Создаем новую папку для аугментированных фотографий
    augmented_image_dir = os.path.join(out_folder, "images")
    os.makedirs(augmented_image_dir, exist_ok=True)

    # Создаем новую папку для новых xml-файлов
    augmented_xml_dir = os.path.join(out_folder, "annotations")
    os.makedirs(augmented_xml_dir, exist_ok=True)

    # Проходимся по всем фотографиям в папке
    for file_name in os.listdir(image_dir):
        if file_name.endswith(".jpg"):
            # Открываем фотографию
            image_path = os.path.join(image_dir, file_name)
            image = Image.open(image_path)

            # Открываем xml-файл с bounding boxes
            xml_path = os.path.join(xml_dir, file_name[:-4] + ".xml")
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Создаем новое имя для аугментированной фотографии
            new_file_name = "flipped_" + file_name
            new_image_path = os.path.join(augmented_image_dir, new_file_name)

            # Переворачиваем фотографию по горизонтали
            flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)

            # Сохраним исходную фотку и исходную аннотацию в той же папке
            image.save(os.path.join(augmented_image_dir, file_name))
            original_tree = ET.ElementTree(root)
            original_tree.write(os.path.join(augmented_xml_dir,
                                             file_name[:-4] + ".xml"))

            # Создаем новый xml-файл с перевернутыми bounding boxes
            for obj in root.findall("object"):
                bbox = obj.find("bndbox")
                xmin = bbox.find("xmin")
                xmax = bbox.find("xmax")
                xmin.text, xmax.text = str(image.width - float(xmax.text)),\
                    str(image.width - float(xmin.text))

            # Сохраняем аугментированный xml-файл
            flipped_tree = ET.ElementTree(root)
            flipped_tree.write(os.path.join(augmented_xml_dir,
                                            "flipped_" + file_name[:-4] + ".xml"))

            # Сохраняем аугментированное изображение
            flipped_image.save(new_image_path)           
    print('Исходное число фотографий и аннотаций =', len(os.listdir(image_dir)))
    print('Итоговое число фотографий и аннотаций =', len(os.listdir(out_folder + '/images')))


def plot_random_image(train_data_loader, hat_class=False):
    '''
    Визуализирует случайную фотографию из датасета, на которой 
    рисует bounding box исходя из xml файла анотации. Так же подписывает
    имя класса над боксом и выбирает цвет отображения разный
    для разных поданных классов
    PS: hat_class=True когда у нас задача двухклассвой детекции
    '''
    import numpy as np
    import cv2
    from  matplotlib import pyplot as plt

    image_batch, label_batch = next(iter(train_data_loader))
    
    image = image_batch[0]
    image = np.transpose(image, (1, 2, 0))
    img = image.numpy().copy() 
    img = (img * 255).astype(np.uint8)
    annot = label_batch[i]

    # Зададим наименования классов:
    if hat_class:
        class_detect = ['none', 'hardhat', 'no_hardhat'] 
    else:
        class_detect = ['none', 'person']

    # Зададим цветовое отображение для классов:
    color_class = [(0,0,255), (0,255,0), (255,50,0)]

    # Пройдемся в цикле по всем боксам на изображении
    for j in range(annot['boxes'].size()[0]):
        [xmin, ymin, xmax, ymax] = annot['boxes'][j]
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        n_class = int(annot['labels'][j])
        text = class_detect[n_class] 
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color_class[n_class], 2)
        img = cv2.putText(img, text, (xmin, ymin - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, color_class[n_class], 3)
    plt.imshow(img)
    plt.show()