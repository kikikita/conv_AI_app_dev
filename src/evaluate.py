import json
import yaml
import pickle
import pandas as pd

from loguru import logger
from sklearn.metrics import roc_auc_score, precision_score,\
    recall_score, f1_score

logger.info('### EVALUATING MODULE ###')
params = yaml.safe_load(open('params.yaml'))['evaluate']
params_train = yaml.safe_load(open('params.yaml'))['train']


def evaluate(model, matrix, logs_path, filename):
    labels = matrix[:, 0].toarray().astype(int)
    x = matrix[:, 1:]

    logger.info('Predicting...')
    probas = model.predict_proba(x)[:, 1]
    predictions = model.predict(x)

    logger.info('Calculating metrics...')
    metrics = {"precision_macro": precision_score(labels, predictions,
                                                  average='macro'),
               "recall_macro": recall_score(labels, predictions,
                                            average='macro'),
               "f1_macro": f1_score(labels, predictions, average='macro'),
               "roc_auc": roc_auc_score(labels, probas)
               }

    logger.info('Metrics saving...')
    with open(logs_path + filename, 'w') as fp:
        json.dump(metrics, fp, indent=4)

    return metrics


model_path = params.get('model_path')
model_tuned_path = params.get('model_tuned_path')
train_data_path = params_train.get('train_data_path')
test_data_path = params.get('test_data_path')
logs_path = params_train.get('logs_path', None)

with open(logs_path + model_path, "rb") as fd:
    model = pickle.load(fd)

with open(logs_path + model_tuned_path, "rb") as fd:
    model_tuned = pickle.load(fd)

with open(logs_path + train_data_path, "rb") as fd:
    train = pickle.load(fd)

with open(logs_path + test_data_path, "rb") as fd:
    test = pickle.load(fd)

logger.info('Models evaluating...')
train_metrics_without_tuning = evaluate(model, train, logs_path,
                                        "train_metrics_without_tuning.json")
test_metrics_without_tuning = evaluate(model, test, logs_path,
                                       "test_metrics_without_tuning.json")
train_metrics_with_tuning = evaluate(model_tuned, train, logs_path,
                                     "train_metrics_with_tuning.json")
test_metrics_with_tuning = evaluate(model_tuned, test, logs_path,
                                    "test_metrics_with_tuning.json")

logger.info('Models comparing...')
metrics_before_tuning = pd.DataFrame([train_metrics_without_tuning,
                                     test_metrics_without_tuning],
                                     index=['train', 'test']).T
metrics_after_tuning = pd.DataFrame([train_metrics_with_tuning,
                                    test_metrics_with_tuning],
                                    index=['train', 'test']).T
metrics_difference = metrics_after_tuning - metrics_before_tuning

logger.info('Saving compared metrics...')
metrics_before_tuning.to_csv(logs_path + 'metrics_before_tuning.csv', sep=';')
metrics_after_tuning.to_csv(logs_path + 'metrics_after_tuning.csv', sep=';')
metrics_difference.to_csv(logs_path + 'metrics_difference.csv', sep=';')
