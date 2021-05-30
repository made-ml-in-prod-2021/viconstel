## Машинное обучение в продакшене
### Домашнее задание №2 

Автор: [viconstel](https://data.mail.ru/profile/k.elizarov/)

1. Установите необходимые пакеты из файла `homework1/requirements.txt`
2. Проект основан на задаче классификации [Heart Disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci)

В рамках данного домашнего задания обученная модель
 `LogisticRegression` из предыдущего ДЗ обернута
 в REST-сервис на основе FastAPI для использования
 на inference в режиме онлайн. <br>
 Доступные эндпоинты:
 ```
/ - дефолтный эндпоинт.
/predict - запрос предсказания модели. Данные передаются
с GET-запросом в формате JSON. 
/docs - документация, сгенерированная FastAPI.
/health - проверка состояния модели на активность.
```

### Описание проекта
Перейдите в директорию домашнего задания №2 `cd /online_inference`
Исходный код REST-сервиса расположен в файле **app.py**.
Тесты сервиса расположены в файле **test/test_app.py**. 
Для запуска тестов используйте команду:
```
pytest tests/test_app.py -v
```
В файле **make_request.py** расположен скрипт для опроса
данного сервиса. Запросы основаны на файле с данными 
из предыдущего домашнего задания `../homework1/data/heart.csv`.
В файле **Dockerfile** описаны слои для формирования 
docker-образа сервиса. Для сборки образа выполните команду:
```
docker build -t viconstel/online_inference:v1 -f Dockerfile ..
```
Образ опубликован на [Dockerhub](https://hub.docker.com/):
```
docker push viconstel/online_inference:v1
```
Для его получения выполните команду:
```
docker pull viconstel/online_inference:v1
```
Для запуска контейнера выполните команду:
```
docker run --name online_inference_v1 -p 8000:8000 viconstel/online_inference:v1
```
Для проверки работы контейнера выполните скрипт:
```
python make_request.py
```

### Самооценка
```
0. Сделано
1. Сделано +3 балла
2. Сделано +3 балла
3. Сделано +2 балла
4. Не сделано
5. Сделано +4 балла
6. В образ копировал только необходимые файлы. Редко меняющиеся образы расположены выше. 
   На основе статьи (https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
   других подходящих этапов оптимизации не выделил. Баллы в данном пункте на ваше усмотрение +0-3 балла
7. Сделано +2 балла
8. Сделано +1 балл
9. Сделано +1 балл
10. Сделано
Сумма: 16 (+0-3 на ваше усмотрение за пункт 6)
```