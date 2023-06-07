import pickle
import json

import optuna
import numpy as np
import yaml
from loguru import logger
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from optuna.integration import CatBoostPruningCallback

logger.info('### TRAINING MODULE ###')
params = yaml.safe_load(open('params.yaml'))['train']

train_data_path = params.get('train_data_path', 'train_data.pkl')
val_data_path = params.get('val_data_path', 'val_data.pkl')
model_name = params.get('model_name', 'model.pkl')
hyperparams = params.get('hyperparams', None)
logs_path = params.get('logs_path', None)


logger.info('Loading prepared data...')
with open(logs_path + train_data_path, 'rb') as fd:
    train_matrix = pickle.load(fd)
with open(logs_path + val_data_path, 'rb') as fd:
    val_matrix = pickle.load(fd)

x_train = train_matrix[:, 1:]
y_train = np.squeeze(train_matrix[:, 0].toarray())

x_val = val_matrix[:, 1:]
y_val = np.squeeze(val_matrix[:, 0].toarray())

logger.info('Model creating and fitting...')
model = CatBoostClassifier(random_state=42)
model.fit(x_train, y_train)

logger.info('Saving trained model...')
with open(logs_path + model_name + '.pkl', "wb") as fd:
    pickle.dump(model, fd)


def objective(trial: optuna.Trial) -> float:
    train_x, valid_x, train_y, valid_y = x_train, x_val, y_train, y_val

    param = {k: eval(v, {'trial': trial}) for k, v in hyperparams.items()}

    gbm = CatBoostClassifier(**param, eval_metric="AUC")

    pruning_callback = CatBoostPruningCallback(trial, "AUC")
    gbm.fit(
        train_x,
        train_y,
        eval_set=[(valid_x, valid_y)],
        verbose=0,
        early_stopping_rounds=100,
        callbacks=[pruning_callback],
    )

    pruning_callback.check_pruned()
    preds = gbm.predict_proba(valid_x)
    roc_auc = roc_auc_score(valid_y, preds[:, 1])

    return roc_auc


n_trials = params.get('n_trials', 100)
timeout = params.get('timeout', 600)

study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        direction="maximize"
    )
study.optimize(objective, n_trials, timeout)
print("Number of finished trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
best_params = study.best_trial.params
print(best_params)

logger.info('Saving best hyperparameters...')
with open(logs_path + 'best_params.json', 'w') as fp:
    json.dump(best_params, fp)

logger.info('Model fitting with best hyperparameters...')
tuned_model = CatBoostClassifier(**best_params, random_state=42)
tuned_model.fit(x_train, y_train)

logger.info('Saving trained with best hyperparameters model...s')
with open(logs_path + model_name + '_tuned' + '.pkl', "wb") as fd:
    pickle.dump(tuned_model, fd)
