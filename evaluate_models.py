from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report

from tf_model import Mimic3Lstm
from tf_gpt2_model import MimicGpt2

import fire
import numpy as np
import os
import pickle
import sys
import tensorflow as tf

MODEL_NAME = 'model_{}'
TARGETS = set([
    'MI',
    'SEPSIS',
    'VANCOMYCIN',
])
N_FEATURES = {
    'MI': 221,
    'SEPSIS': 225,
    'VANCOMYCIN': 224,
}
N_ATTN_HEADS = {
    'MI': 13,
    'SEPSIS': 9,
    'VANCOMYCIN': 8,
}
PICKLED_OBJECTS = {
    'x_train': 'X_TRAIN_{}.txt',
    'y_train': 'Y_TRAIN_{}.txt',
    'x_val': 'X_VAL_{}.txt',
    'y_val': 'Y_VAL_{}.txt',
    'x_test': 'X_TEST_{}.txt',
    'y_test': 'Y_TEST_{}.txt',
}


def evaluate(archi='lstm', models_dir='saved_models', pickled_dir='pickled_objects'):
    # check saved_models directory existence
    if not os.path.isdir(models_dir):
        print(f'[ERROR] Model directory "{models_dir}" does not exist.')
        print(f'[ERROR] Run "rnn_mimic.py" before this script.')
        sys.exit(1)

    # check pickled_objects existence
    if not os.path.isdir(pickled_dir):
        print(f'[ERROR] Pickled directory "{pickled_dir}" does not exist.')
        print(f'[ERROR] Run "rnn_mimic.py" before this script.')
        sys.exit(1)

    # check models existence
    models = os.listdir(models_dir)
    if not models:
        print(f'[ERROR] No saved models in "{models_dir}".')
        print(f'[ERROR] Run "rnn_mimic.py" before this script.')
        sys.exit(1)

    # retrieve data for the specific target
    for target in TARGETS:
        model_name = MODEL_NAME.format(target)
        model_path = os.path.join(models_dir, f'weights_{target}', model_name)

        # check for model existence
        if not os.path.isfile(f'{model_path}.index'):
            print(f'[ERROR] {model_path} is not found')
            sys.exit(1)

        # retrieve data for this model
        data = get_data(target=target)

        # load saved model
        model = get_model(archi, target)
        model.load_weights(model_path).expect_partial()

        # prepare inputs and mask
        x_test = data['x_test'].astype(np.float32)
        y_boolmat = np.reshape(np.any(x_test, axis=2), (-1, 14, 1))

        # calculate model predictions (for performance evaluation)
        y_pred, _ = model(x_test)
        y_pred = y_pred[y_boolmat]
        y_test = data['y_test'][y_boolmat]

        # output model performance statistics
        cm = confusion_matrix(y_test, np.around(y_pred))
        acc = accuracy_score(y_test, np.around(y_pred))
        auc = roc_auc_score(y_test, y_pred)
        print(f'\n[INFO] Evaluating model for {target}')
        print(f'    Confusion Matrix Validation')
        print(cm)
        print(f'    Validation Accuracy: {acc:.4%}')
        print(f'    ROC AUC SCORE VAL: {auc:.4%}')
        print('    CLASSIFICATION REPORT VAL')
        print(classification_report(y_test, np.around(y_pred)))
        print('=' * 40)


def get_model(archi, target):
    if archi == 'lstm':
        n_features = N_FEATURES[target]
        return Mimic3Lstm(n_features)

    if archi == 'gpt2':
        n_features = N_FEATURES[target]
        n_attn_heads = N_ATTN_HEADS[target]
        return MimicGpt2(n_features, n_attn_heads)

    raise AssertionError(f'Unknown model "{archi}"')


def get_data(*, target, pickled_dir='pickled_objects'):
    if target not in TARGETS:
        print(f'[ERROR] Invalid target: {target}')
        sys.exit(1)

    result = {}
    for k, v in PICKLED_OBJECTS.items():
        # build complete pickled filename (including target)
        fpath = os.path.join(pickled_dir, v.format(target))

        # check its existence
        if not os.path.isfile(fpath):
            print(f'[ERROR] {fpath} does not exist.')
            sys.exit(1)

        # store contents to dictionary
        with open(fpath, 'rb') as fd:
            result[k] = pickle.load(fd)

    return result


if __name__ == '__main__':
    fire.Fire(evaluate)
