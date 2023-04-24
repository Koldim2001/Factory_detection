# train - код, реализующий обучение нейронной сети

def train(model, train_data_loader, optimizer,
          val_data_loader, save_path, num_epochs=2, comment='test', device='cpu'):
    
    import torch, torchvision
    from tqdm import tqdm
    from torch.utils.tensorboard import SummaryWriter

    # Инициализация Tensorboard
    writer = SummaryWriter(comment=comment)
    
    model.to(device)
    # Задал значение стартового наивысщего лосса валидации:
    best_loss = float('inf') 

    # Цикл по эпохам обучения:
    for epoch in range(num_epochs):
        
        print(f'Идет обучение {epoch+1} эпохи (из {num_epochs})')   

        # Будем считать суммарную ошибку на эпохе, поэтому первоначально занулим лоссы
        epoch_loss = 0
        loss_classifier = 0
        loss_box_reg = 0
        loss_objectness = 0
        loss_rpn_box_reg = 0

        # Переведем модель в режим обучения и начнем итераыии по батчам
        model.train()
        for i, (images, targets) in tqdm(enumerate(train_data_loader)):
            
            # Стопаю итерацию по батчам после 25 спусков. Из-за suffle=True фотки каждый раз новые
            # Благодаря этому мы сможем чаще оценивать loss валидации 
            if i == 25:
                break

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Получим словарь со значниями loss функций :
            loss_dict = model(images, targets)       

            # Найдем суммарный loss и сделаем шаг град. спуска:
            losses = sum(loss for loss in loss_dict.values())       
            optimizer.zero_grad()
            losses.backward()
            optimizer.step() 

            # Используем для подсчета всехсредних лоссов за эпоху:
            loss_classifier += loss_dict['loss_classifier'].item()
            loss_box_reg += loss_dict['loss_box_reg'].item()
            loss_objectness += loss_dict['loss_objectness'].item()
            loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()
            epoch_loss += losses

        # Запишем в TensorBoard значения лоссов на трейне:    
        writer.add_scalar('Summ train loss during epochs', epoch_loss/i, epoch+1)
        writer.add_scalar('Train loss_classifier', loss_classifier/i, epoch+1)
        writer.add_scalar('Train loss_box_reg', loss_box_reg/i, epoch+1)
        writer.add_scalar('Train loss_objectness', loss_objectness/i, epoch+1)
        writer.add_scalar('Train loss_rpn_box_reg', loss_rpn_box_reg/i, epoch+1)

        print(f'Train summ loss after {epoch+1} epochs = {epoch_loss/i}')

        # Обнулим суммарный лосс валидации
        val_total_loss = 0
        # Перевод модели в режим валидации для оценки суммарного лосса на вал. датасете
        with torch.no_grad():
            for images, targets in tqdm(val_data_loader):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets) 

                # Вычислим лосс на батче
                losses = sum(loss for loss in loss_dict.values())        
                val_total_loss += losses.item()

            # Запишем в TensorBoard значение суммарного лосса на валидации:
            writer.add_scalar('Summ validation loss during epochs',
                            val_total_loss/len(val_data_loader), epoch+1)
            print(f'Validation summ loss after {epoch+1} epochs = {val_total_loss/len(val_data_loader)}')

        '''
        Если на валидации мы получили меньше лосс чем текущий best_loss, то сохраняем веса модели
        Тем самым мы на выходе получим модель, имеющую самый низкий лосс на валидации, 
        из всех обученных нами эпох
        '''
        if val_total_loss/len(val_data_loader) < best_loss:
            state_dict = model.state_dict()
            print('Сохраним новую модель, так как текущая конфигурация имеет ниже val loss')
            best_loss = val_total_loss/len(val_data_loader)
            torch.save(state_dict, save_path)
            
    # Закрытие Tensorboard
    writer.close()   
    print('Обучение завершено')

