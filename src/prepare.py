import pandas as pd
import scipy.sparse as sparse
import yaml
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from loguru import logger

logger.info('### PREPARING MODULE ###')
params = yaml.safe_load(open('params.yaml'))['prepare']
params_train = yaml.safe_load(open('params.yaml'))['train']

data_path = params.get('data', None)
logs_path = params_train.get('logs_path', None)

logger.info('Receiving the data...')
df = pd.read_csv(data_path, on_bad_lines='skip', sep=';')
target_col = params.get('target_col', 'isHate')
text_col = params.get('text_col', 'comment')
df[target_col] = df[target_col].apply(lambda x: 1 if x > 0.5 else 0)
df_train, df_test = train_test_split(df, random_state=42,
                                     stratify=df[target_col],
                                     train_size=0.7)
df_val, df_test = train_test_split(df_test, random_state=42,
                                   stratify=df_test[target_col],
                                   train_size=0.5)

max_features = params.get('max_features', 2000)
word_vect = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    stop_words='english',
    ngram_range=(1, 2),
    max_features=max_features)

logger.info('Preparing the data...')
word_vect.fit(df['comment'])
train_word_features = word_vect.transform(df_train['comment'])
val_word_features = word_vect.transform(df_val['comment'])
test_word_features = word_vect.transform(df_test['comment'])

X_train = train_word_features.tocsr()
X_val = val_word_features.tocsr()
X_test = test_word_features.tocsr()


def save_matrix(df, matrix, output):
    label_matrix = sparse.csr_matrix(df[target_col]).T
    result = sparse.hstack([label_matrix, matrix], format="csr")
    with open(output, "wb") as fd:
        pickle.dump((result), fd)


logger.info('Saving prepared data...')
save_matrix(df_train, X_train, logs_path + 'train_data.pkl')
save_matrix(df_val, X_val, logs_path + 'val_data.pkl')
save_matrix(df_test, X_test, logs_path + 'test_data.pkl')
