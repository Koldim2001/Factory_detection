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
