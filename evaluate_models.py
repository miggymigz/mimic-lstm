from keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report

import fire
import numpy as np
import os
import pickle
import sys

MODEL_NAME = 'kaji_mach_final_no_mask_{}_pad14.h5'
TARGETS = set([
    'MI',
    'SEPSIS',
    'VANCOMYCIN',
])
PICKLED_OBJECTS = {
    'x_train': 'X_TRAIN_{}.txt',
    'y_train': 'Y_TRAIN_{}.txt',
    'x_val': 'X_VAL_{}.txt',
    'y_val': 'Y_VAL_{}.txt',
    'x_boolmat_val': 'x_boolmat_val_{}.txt',
    'y_boolmat_val': 'y_boolmat_val_{}.txt',
    'no_feature_cols': 'no_feature_cols_{}.txt',
}


def evaluate(models_dir='saved_models', pickled_dir='pickled_objects'):
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
        model_path = os.path.join(models_dir, model_name)

        # check for model existence
        if not os.path.isfile(model_path):
            print(f'[ERROR] {model_path} is not found')
            sys.exit(1)

        # retrieve data for this model
        data = get_data(target=target)

        # load saved model
        model = load_model(model_path)

        y_pred = model.predict(data['x_val'])
        y_pred = y_pred[~data['y_boolmat_val']]
        np.unique(y_pred)
        y_val = data['y_val'][~data['y_boolmat_val']]
        y_pred_train = model.predict(data['x_train'])

        # output model performance statistics
        cm = confusion_matrix(y_val, np.around(y_pred))
        acc = accuracy_score(y_val, np.around(y_pred))
        auc = roc_auc_score(y_val, y_pred)
        print(f'\n[INFO] Evaluating model for {target}')
        print(f'    Confusion Matrix Validation')
        print(cm)
        print(f'    Validation Accuracy: {acc:.4%}')
        print(f'    ROC AUC SCORE VAL: {auc:.4%}')
        print('    CLASSIFICATION REPORT VAL')
        print(classification_report(y_val, np.around(y_pred)))
        print('=' * 40)


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
