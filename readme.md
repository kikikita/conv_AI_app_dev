# HW1. Experiment management
## Задача

Провести эксперимент с обучением модели, используя такие инструменты для оптимизации ML-моделей и отслеживания экспериментов, как Optuna, Hyperopt, Pandas, Polars и DVC.

## Dataset

Ethos_Dataset_Binary.csv - набор данных для обнаружения разжигающих ненависть высказываний на платформах социальных сетей, содержит 998 комментариев в наборе данных вместе с меткой о наличии или отсутствии разжигающих ненависть высказываний. 565 из них не содержат разжигания ненависти, в то время как остальные, 433, содержат.
Целевая колонка «isHate» изначально содержит значения между 0 и 1, в дальнейшем данные значения преобразованы в 0 или 1, с использованием порогового значения равного 0.5.

После преобразования данных в колонке «comment», датасет был разбит на train, val и test части с размерами 0.7, 0.15, 0.15 от исходного соответственно.

## Model Selection

Для извлечения признаков был использован TfidfVectorizer из библиотеки sklearn с параметрами ngram_range = (1, 2) и max_features = 2000.
Для базовой модели был выбран классификатор CatBoostClassifier из библиотеки catboost со значением параметра random_state = 42.
 
## Hyperparameter Optimization

Для оптимизации гиперпараметров была выбрана библиотека Optuna. В методе n_trials установлены следующие параметры: 100 для количества проб, 600 для временного ограничения, 5 для количества шагов разогрева, и "maximize" для направления оптимизации метрики "AUC".
Ниже приведены поля для поиска гиперпараметров:
•	iterations: (trial.suggest_int("iterations", 1000, 3000)) - количество итераций
•	learning_rate: (trial.suggest_float("learning_rate", 0.01, 0.5, log=True)) - шаг обучения
•	colsample_bylevel: (trial.suggest_float("colsample_bylevel", 0.01, 0.1, log=True)) - метод случайного подпространства
•	l2_leaf_reg: (trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True)) - коэффициент в члене регуляризации L2 функции затрат
•	depth: (trial.suggest_int("depth", 1, 12)) - глубина дерева

## Experiment Tracking

Для логирования датасета и результатов экспериментов используется библиотека dvc с дальнейшим сохранением на Google Drive. Кроме того, все модели, найденные гиперпараметры, тренировочные и тестовые данные, а также метрики сохраняются в файлы JSON (в папку logs) и логируются при помощи библиотеки logger.
Code
Код эксперимента хранится в файлах с расширением «.py»: prepare.py (подготовка данных), train.py (обучение модели и подбор гиперпараметров), evaluate.py (сравнение моделей и сохранение метрик). В файле params.yaml находятся параметры для запуска скриптов. 
Пайплайн обучения запускается скриптом launch_sc.sh, либо launch_sc.bat (для windows) которые находится в папке с кодом.
Помимо «.py» файлов, процесс предобработки данных, обучения и оценки моделей представлен в ноутбуке hw1.ipynb.
