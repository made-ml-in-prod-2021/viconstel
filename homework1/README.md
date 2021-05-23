## Машинное обучение в продакшене
### Домашнее задание №1 

Автор: [viconstel](https://data.mail.ru/profile/k.elizarov/)

1. Установите необходимые пакеты из файла `requirements.txt`
2. Проект основан на задаче классификации [Heart Disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci)
3. В качестве классфикаторов были использваны модели `LogisticRegression` и 
`KNeighborsClassifier`. Параметры для выбранной модели классификации можно настроить
в соответствующих файлах:
```
logreg_config.yml

train_params:
  model_type: "LogisticRegression"
  penalty: "l2"
  C: 1.0
  fit_intercept: True
  solver: "lbfgs"
  max_iter: 1300
  random_state: 111
```
```
knn_config.yml

train_params:
  model_type: "KNeighborsClassifier"
  n_neighbors: 5
  algorithm: "auto"
  metric: "minkowski"
```
Также в файлах конфигурации необходимо указать настройки логирования и
 настроить пути для файлов с данными и пути
для файлов, в которых будут сохранены артефакты модели после обучения:
```
# Файл с данными для обучения
input_train_data_path: "data/heart.csv"
# Файл с данными для предсказания модели
input_test_data_path: "data/test_sample.csv"
# Файл с обученной моделью
output_model_path: "models/model.pkl"
# Файл с обученным пайплайном препроцессинга данных
output_preprocessor_path: "models/preprocessor.pkl"
# Файл с метриками на валидационном сете
metric_path: "models/metrics.json"
# Файл, в который будут записаны предсказания модели
predictions_path: "data/predictions.csv"
# Файл конфигурации логирования
logging_config: "configs/logging_conf.yml"
# Настройки разбиения на обучающую и валидационную выборку
split_params:
  val_size: 0.2
  random_state: 3
# Категориальные, числовые поля в данных и целевое поле
feature_params:
  categorical_features:
    - "sex"
    - "cp"
    ...
  numerical_features:
    - "age"
    - "trestbps"
    ...
  target: "target"
```

Для запуска обучения логистической регрессии в режиме обучения или валидации
выполните следующие команды с соответствующими аргументами:
```
python ml_project\run_pipeline.py configs\logreg_config.yml train 
```
```
python ml_project\run_pipeline.py configs\logreg_config.yml val
```
Для запуска обучения модели k-NN в режиме обучения или валидации
выполните следующие команды с соответствующими аргументами:
```
python ml_project\run_pipeline.py configs\knn_config.yml train 
```
```
python ml_project\run_pipeline.py configs\knn_config.yml val
```
Возможен запуск без аргументов (дефолтные значения: `config_path=configs\logreg_config.yml`,
`train_val=train`)
```
python ml_project\run_pipeline.py
```
При запуске с аргументом `train` модель обучится и соответствующие артефакты обучения
(модель,
обученный пайплайн препроцессинга данных и JSON-файл с метриками)
 будут размещены в директории `models`. Предсказания модели 
(запуск с аргументом `val`) помещаются в файл `data/predictions.csv`.

<br><br>

В качестве препроцессинга категориальных признаков используются:
```
1. SimpleImputer со стратегией `median`
2. OneHotEncoder
```
Для числовых признаков используются:
```
1. SimpleImputer со стратегией `mean`
2. Кастомный трансформер CustomStandardScaler (файл /ml_project/features/build_features.py)
```
В качестве метрик оценки качества модели используются:
```
1. Accuracy score
2. F1 score
```

## Структура проекта
```
+ configs - директория с файлами конфигурации
    + knn_config.yml - настройка KNeighborsClassifier
    + logreg_config.yml - настройка LogisticRegression
    + logging_conf.yml - настройка логирования
+ data - директория с данными
    + heart.csv - данные для обучения
    + test_sample.csv - данные для тестирования
+ ml_project - директория с исходным кодом
    + data
        + make_dataset.py - чтение и разбиение обучающих данных
    + entities - директория сущностей из файлов конфигурации
        + feature_params.py
        + split_params.py
        + pipeline_params.py
        + train_model_params.py
    + features
        + build_features.py - файл препроцессинга данных
    + models
        + model.py - файл с моделью классификации
    + run_pipeline.py - файл запуска приложения с соответствующим пайплайном
+ models - директория для артефактов модели
+ notebooks - директория для Jupyter ноутбуков 
    + 1.0-ekv-initial-data-exploration.ipynb - Jupyter Notebook with EDA
+ tests - директория с тестами
    + test_custom_scaler.py - тестирование CustomStandardScaler
+ README.md - файл описания (этот файл)
+ requirements.txt - файл с необходимыми пакетами
+ setup.py - файл пакетирования
```

### Самопроверка
```
-2. Сделано +1 балл 
-1. Сделано 
0. Сделано +2 балла 
1. Сделано +2 балла 
2. Сделано +2 балла 
3. Сделано +2 балла 
4. Не сделано 
5. Не сделано 
6. Сделано +3 балла 
7. Сделано +3 балла 
8. Сделано +3 балла. Кастомный трансформер протестирован 
9. Сделано +3 балла 
10. Сделано +3 балла 
11. Не сделано 
12. Не сделано 
13. Сделано +1 балл 
Суммарно: 25 баллов
```