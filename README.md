# Проект по детекции
Проект по созданию системы безопасности на предприятии, реализующей детектирование людей и автоматическую проверку наличия касок по изображениям с видеокамер

---

### Как запускать программу:
(данные команды требуется запускать в терминале)
1. Склонируйте к себе этот репозиторий 
```
git clone https://github.com/Koldim2001/Factory_detection.git
```
2. Gерейдите с помощью команды cd в созданную папку Factory_detection
```
cd Factory_detection
```
3. Загрузите все необходимые библиотеки:
```
pip install -r requirements.txt
```
4. Запустите написанный python скрипт:
```
python detecting.py
```

При запуске программы потребуется ввести путь к изображению, для которого надо провети детекцию.
После завершения детектирования людей откроется отдельное окно с результирующими боксами. При закрытии этого окна
автоматически начнется процедура двухклассовой детекции (наличие/отсутвие касок на голове). По результатам вычислений откроется отдельное окно с задетектированными боксами.<br><br>
_Примеры результатов работы двух разных обученных моделей:_

<img align="center" src="https://drive.google.com/uc?id=1Dtu_bK9w5Hl65A6lETChuu1Ftz2wirUi" alt="kolesnokov__dima" height="260" width="450" /> <img align="center" src="https://drive.google.com/uc?id=105RsKrPwpzGLTbyUYjKDsDRP0bd6IUIT" alt="kolesnokov__dima" height="260" width="440" /> </center>  



<div style="text-align:center;">
  <img src="https://drive.google.com/uc?id=1Dtu_bK9w5Hl65A6lETChuu1Ftz2wirUi" alt="Alt person detection" width="440" height="250">
  <img src="https://drive.google.com/uc?id=105RsKrPwpzGLTbyUYjKDsDRP0bd6IUIT" alt="Alt hardhat detection" width="440" height="250">
</div>
