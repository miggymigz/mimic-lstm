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
    # 'x_train': 'X_TRAIN_{}.txt',
    # 'y_train': 'Y_TRAIN_{}.txt',
    # 'x_val': 'X_VAL_{}.txt',
    # 'y_val': 'Y_VAL_{}.txt',
    'x_test': 'X_TEST_{}.txt',
    'y_test': 'Y_TEST_{}.txt',
}


def evaluate(
    architecture='lstm',
    layers=4,
    models_dir='saved_models',
    pickled_dir='pickled_objects'
):
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
        # prepare inputs and mask
        data = get_data(target=target)
        x_test = data['x_test'].astype(np.float32)
        y_boolmat = np.reshape(np.any(x_test, axis=2), (-1, 14, 1))

        # load saved model
        batch_size = x_test.shape[0]
        model = get_model(
            architecture=architecture,
            layers=int(layers),
            target=target,
            batch_size=batch_size,
            models_dir=models_dir,
        )

        # calculate model predictions (for performance evaluation)
        y_pred, _ = model(x_test)
        y_pred = y_pred[y_boolmat]
        y_test = data['y_test'][y_boolmat]

        # output model performance statistics
        cm = confusion_matrix(y_test, np.around(y_pred))
        acc = accuracy_score(y_test, np.around(y_pred))
        auc = roc_auc_score(y_test, y_pred)
        print(f'\n[INFO] Evaluating model for {target}')
        print(f'    Confusion Matrix')
        print(cm)
        print(f'    Accuracy: {acc:.4%}')
        print(f'    ROC AUC SCORE: {auc:.4%}')
        print('    CLASSIFICATION REPORT')
        print(classification_report(y_test, np.around(y_pred)))
        print('=' * 40)


def get_model(*, architecture, layers, target, batch_size, models_dir='saved_models'):
    print(f'[INFO] Using architecture={architecture}, layers={layers}')

    model_name = MODEL_NAME.format(target)
    model_path = os.path.join(
        models_dir,
        f'{architecture}_weights_{target}',
        model_name,
    )

    # check for model existence
    if not os.path.isfile(f'{model_path}.index'):
        print(f'[ERROR] {model_path} is not found')
        sys.exit(1)

    if architecture == 'lstm':
        print(f'[INFO] Parameter layers={layers} will not be used')
        n_features = N_FEATURES[target]
        model = Mimic3Lstm(n_features, batch_size=batch_size)
        model.load_weights(model_path)
        model.build(tf.TensorShape([batch_size, 14, n_features]))
        return model

    if architecture == 'gpt2':
        n_features = N_FEATURES[target]
        n_attn_heads = N_ATTN_HEADS[target]
        model = MimicGpt2(n_features, n_attn_heads, n_layers=layers)
        model.load_weights(model_path)
        return model

    raise AssertionError(f'Unknown model "{architecture}"')


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
