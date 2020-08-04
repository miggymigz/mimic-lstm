''' Recurrent Neural Network in Keras for use on the MIMIC-III '''

from pad_sequences import PadSequences
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
from time import time

from models.tf_base_lstm import Mimic3BaseLstm
from models.tf_lstm import Mimic3Lstm
from models.tf_gpt2 import Mimic3Gpt2

import fire
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf

TIMESTEPS = 14
ROOT = os.path.join('mimic_database', 'mapped_elements')
PREPROCESSED_FILE = os.path.join(ROOT, 'CHARTEVENTS_preprocessed.csv')

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

# feature IDs useful for proper masking
ID_MEDS_START = {
    'MI': 147,
    'SEPSIS': 151,
    'VANCOMYCIN': 151,
}


######################################
## MAIN ###
######################################


def wbc_crit(x):
    if (x > 12 or x < 4) and x != 0:
        return 1
    else:
        return 0


def temp_crit(x):
    if (x > 100.4 or x < 96.8) and x != 0:
        return 1
    else:
        return 0


def return_data(
    balancing_scheme='truncate',
    target='MI',
    return_cols=False,
    tt_split=0.7,
    val_percentage=0.8,
    cross_val=False,
    mask=False,
    dataframe=False,
    split=True,
    pad=True,
):
    """

    Returns synthetic or real data depending on parameter

    Args:
    -----
        synth_data : synthetic data is False by default
        balance : whether or not to balance positive and negative time windows 
        target : desired target, supports MI, SEPSIS, VANCOMYCIN or a known lab, medication
        return_cols : return columns used for this RNN
        tt_split : fraction of dataset to use fro training, remaining is used for test
        cross_val : parameter that returns entire matrix unsplit and unbalanced for cross val purposes
        mask : 24 hour mask, default is False
        dataframe : returns dataframe rather than numpy ndarray
        split : creates test train splits
        pad : by default is True, will pad to the time_step value
    Returns:
    -------
        Training and validation splits as well as the number of columns for use in RNN  

    """
    df = pd.read_csv(PREPROCESSED_FILE, low_memory=False)

    if target == 'MI':
        df[target] = ((df['troponin'] > 0.4) & (
            df['CKD'] == 0)).apply(lambda x: int(x))

    elif target == 'SEPSIS':
        df['hr_sepsis'] = df['heart rate'].apply(
            lambda x: 1 if x > 90 else 0)
        df['respiratory rate_sepsis'] = df['respiratory rate'].apply(
            lambda x: 1 if x > 20 else 0)
        df['wbc_sepsis'] = df['WBCs'].apply(wbc_crit)
        df['temperature f_sepsis'] = df['temperature (F)'].apply(temp_crit)
        df['sepsis_points'] = (
            df['hr_sepsis'] + df['respiratory rate_sepsis'] + df['wbc_sepsis'] + df['temperature f_sepsis'])
        df[target] = ((df['sepsis_points'] >= 2) & (
            df['Infection'] == 1)).apply(lambda x: int(x))

        del df['hr_sepsis']
        del df['respiratory rate_sepsis']
        del df['wbc_sepsis']
        del df['temperature f_sepsis']
        del df['sepsis_points']
        del df['Infection']

    elif target == 'PE':
        df['blood_thinner'] = (df['heparin'] + df['enoxaparin'] +
                               df['fondaparinux']).apply(lambda x: 1 if x >= 1 else 0)
        df[target] = (df['blood_thinner'] & df['ct_angio'])
        del df['blood_thinner']

    elif target == 'VANCOMYCIN':
        df['VANCOMYCIN'] = df['vancomycin'].apply(
            lambda x: 1 if x > 0 else 0)
        del df['vancomycin']

    df = df.select_dtypes(exclude=['object'])

    if pad:
        pad_value = 0
        df = PadSequences().pad(df, 1, TIMESTEPS, pad_value=pad_value)
        print('There are {0} rows in the df after padding'.format(len(df)))

    COLUMNS = list(df.columns)

    if target == 'MI':
        toss = ['ct_angio', 'troponin', 'troponin_std',
                'troponin_min', 'troponin_max', 'Infection', 'CKD']
        COLUMNS = [i for i in COLUMNS if i not in toss]
    elif target == 'SEPSIS':
        toss = ['ct_angio', 'Infection', 'CKD']
        COLUMNS = [i for i in COLUMNS if i not in toss]
    elif target == 'PE':
        toss = ['ct_angio', 'heparin', 'heparin_std', 'heparin_min',
                'heparin_max', 'enoxaparin', 'enoxaparin_std',
                'enoxaparin_min', 'enoxaparin_max', 'fondaparinux',
                'fondaparinux_std', 'fondaparinux_min', 'fondaparinux_max',
                'Infection', 'CKD']
        COLUMNS = [i for i in COLUMNS if i not in toss]
    elif target == 'VANCOMYCIN':
        toss = ['ct_angio', 'Infection', 'CKD']
        COLUMNS = [i for i in COLUMNS if i not in toss]

    COLUMNS.remove(target)

    if 'HADM_ID' in COLUMNS:
        COLUMNS.remove('HADM_ID')
    if 'SUBJECT_ID' in COLUMNS:
        COLUMNS.remove('SUBJECT_ID')
    if 'YOB' in COLUMNS:
        COLUMNS.remove('YOB')
    if 'ADMITYEAR' in COLUMNS:
        COLUMNS.remove('ADMITYEAR')

    if return_cols:
        return COLUMNS

    if dataframe:
        return (df[COLUMNS+[target, "HADM_ID"]])

    MATRIX = df[COLUMNS+[target]].values
    MATRIX = MATRIX.reshape(
        int(MATRIX.shape[0]/TIMESTEPS),
        TIMESTEPS,
        MATRIX.shape[1]
    )

    # keep a copy of the original dataset
    ORIG_MATRIX = np.copy(MATRIX)

    # note we are creating a second order bool matrix
    # create a mask which will be true for padding days
    # turn padding values from zeroes to nan
    # normalize data and return the mean and std used
    bool_matrix = (~MATRIX.any(axis=2))
    MATRIX[bool_matrix] = np.nan
    MATRIX, means, stds = PadSequences().ZScoreNormalize(MATRIX)

    # restore 3D shape to boolmatrix for consistency
    bool_matrix = np.isnan(MATRIX)
    MATRIX[bool_matrix] = pad_value

    permutation = np.random.permutation(MATRIX.shape[0])
    MATRIX = MATRIX[permutation]
    bool_matrix = bool_matrix[permutation]

    # also reshuffle the original matrix
    ORIG_MATRIX = ORIG_MATRIX[permutation]

    X_MATRIX = MATRIX[:, :, 0:-1]
    Y_MATRIX = MATRIX[:, :, -1]

    x_bool_matrix = bool_matrix[:, :, 0:-1]
    y_bool_matrix = bool_matrix[:, :, -1]

    X_TRAIN = X_MATRIX[0:int(tt_split*X_MATRIX.shape[0]), :, :]
    Y_TRAIN = Y_MATRIX[0:int(tt_split*Y_MATRIX.shape[0]), :]
    Y_TRAIN = Y_TRAIN.reshape(Y_TRAIN.shape[0], Y_TRAIN.shape[1], 1)

    # save a copy of the unnormalized training set
    ORIG_TRAIN = ORIG_MATRIX[:int(tt_split*X_MATRIX.shape[0])]

    X_VAL = X_MATRIX[int(tt_split*X_MATRIX.shape[0])                     :int(val_percentage*X_MATRIX.shape[0])]
    Y_VAL = Y_MATRIX[int(tt_split*Y_MATRIX.shape[0])                     :int(val_percentage*Y_MATRIX.shape[0])]
    Y_VAL = Y_VAL.reshape(Y_VAL.shape[0], Y_VAL.shape[1], 1)

    # save a copy of the unnormalized validation set
    ORIG_VAL = ORIG_MATRIX[int(
        tt_split*X_MATRIX.shape[0]):int(val_percentage*X_MATRIX.shape[0])]

    x_val_boolmat = x_bool_matrix[int(
        tt_split*x_bool_matrix.shape[0]):int(val_percentage*x_bool_matrix.shape[0])]
    y_val_boolmat = y_bool_matrix[int(
        tt_split*y_bool_matrix.shape[0]):int(val_percentage*y_bool_matrix.shape[0])]
    y_val_boolmat = y_val_boolmat.reshape(
        y_val_boolmat.shape[0], y_val_boolmat.shape[1], 1)

    X_TEST = X_MATRIX[int(val_percentage*X_MATRIX.shape[0])::]
    Y_TEST = Y_MATRIX[int(val_percentage*X_MATRIX.shape[0])::]
    Y_TEST = Y_TEST.reshape(Y_TEST.shape[0], Y_TEST.shape[1], 1)

    # save a copy of the unnormalized test set
    ORIG_TEST = ORIG_MATRIX[int(val_percentage*X_MATRIX.shape[0])::]

    x_test_boolmat = x_bool_matrix[int(
        val_percentage*x_bool_matrix.shape[0])::]
    y_test_boolmat = y_bool_matrix[int(
        val_percentage*y_bool_matrix.shape[0])::]
    y_test_boolmat = y_test_boolmat.reshape(
        y_test_boolmat.shape[0], y_test_boolmat.shape[1], 1)

    X_TEST[x_test_boolmat] = pad_value
    Y_TEST[y_test_boolmat] = pad_value

    # balance dataset samples
    X_TRAIN, Y_TRAIN = balance_set(X_TRAIN, Y_TRAIN, scheme=balancing_scheme)
    X_VAL, Y_VAL = balance_set(X_VAL, Y_VAL, scheme=balancing_scheme)
    X_TEST, Y_TEST = balance_set(X_TEST, Y_TEST, scheme=balancing_scheme)

    no_feature_cols = X_TRAIN.shape[2]

    if mask:
        print('MASK ACTIVATED')
        X_TRAIN = np.concatenate(
            [np.zeros((X_TRAIN.shape[0], 1, X_TRAIN.shape[2])), X_TRAIN[:, 1::, ::]], axis=1)
        X_VAL = np.concatenate(
            [np.zeros((X_VAL.shape[0], 1, X_VAL.shape[2])), X_VAL[:, 1::, ::]], axis=1)

    if cross_val:
        return (MATRIX, no_feature_cols)

    if split:
        norm_params = {
            'means': means,
            'stds': stds,
        }
        orig_data = {
            'train': ORIG_TRAIN,
            'val': ORIG_VAL,
            'test': ORIG_TEST,
        }

        return (
            X_TRAIN, Y_TRAIN,
            X_VAL, Y_VAL,
            X_TEST, Y_TEST,
            norm_params, orig_data,
        )
    else:
        return (
            np.concatenate((X_TRAIN, X_VAL), axis=0),
            np.concatenate((Y_TRAIN, Y_VAL), axis=0),
            no_feature_cols,
        )


def balance_set(x, y, scheme='duplicate'):
    dataset = np.concatenate([x, y], axis=2)
    pos_ind = np.unique(np.where((dataset[:, :, -1] == 1).any(axis=1))[0])
    neg_ind = np.unique(np.where(~(dataset[:, :, -1] == 1).any(axis=1))[0])

    # shuffle indices of positive and negative samples
    np.random.shuffle(pos_ind)
    np.random.shuffle(neg_ind)

    positive_samples_count = pos_ind.shape[0]
    negative_samples_count = neg_ind.shape[0]

    if scheme == 'duplicate':
        multiplier = negative_samples_count // positive_samples_count
        remainder = negative_samples_count % positive_samples_count
        total_ind = np.hstack([
            pos_ind.tolist() * multiplier,  # add more positive samples
            pos_ind[:remainder],  # add a bit more for perfection
            neg_ind,  # retain all negative samples
        ])
    elif scheme == 'truncate':
        length = min(positive_samples_count, negative_samples_count)
        total_ind = np.hstack([pos_ind[0:length], neg_ind[0:length]])
    else:
        raise AssertionError(f'Unknown balancing scheme: {scheme}')

    # shuffle indices of the combined samples
    np.random.shuffle(total_ind)
    ind = total_ind

    x_balanced = dataset[ind, :, 0:-1]
    y_balanced = dataset[ind, :, -1]
    y_balanced = y_balanced.reshape(
        y_balanced.shape[0],
        y_balanced.shape[1],
        1,
    )

    return x_balanced, y_balanced


def create_model(architecture, target, n_layers=1):
    n_features = N_FEATURES[target]
    n_attn_heads = N_ATTN_HEADS[target]
    n_layers = n_layers if isinstance(n_layers, int) else int(n_layers)

    if architecture == 'base':
        return Mimic3BaseLstm(time_steps=TIMESTEPS)

    if architecture == 'lstm':
        return Mimic3Lstm(time_steps=TIMESTEPS)

    if architecture == 'gpt2':
        return Mimic3Gpt2(
            n_features,
            n_attn_heads,
            n_days=TIMESTEPS,
            n_layers=n_layers,
        )

    raise AssertionError(f'ERROR - Unknown architecture="{architecture}"')


def create_optimizer(optimizer, lr=0.001):
    if optimizer == 'rmsprop':
        return tf.keras.optimizers.RMSprop(
            learning_rate=lr,
            rho=0.9,
            epsilon=1e-08,
        )

    if optimizer == 'adam':
        return tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9,
        )

    raise AssertionError(f'ERROR - Unknown optimizer="{optimizer}"')


def train(
    target='MI',
    model_name="kaji_mach_0",
    balancer=True,
    evaluate=False,
    architecture='lstm',
    epochs=10,
    optimizer='rmsprop',
    layers=1,
    batch_size=16,
):
    """

    Use Keras model.fit using parameter inputs

    Args:
    ----
    model_name : Parameter used for naming the checkpoint_dir
    synth_data : Default to False. Allows you to use synthetic or real data.

    Return:
    -------
    Nonetype. Fits model only. 

    """

    print(f'[train] START: model_name={model_name}, target={target}')

    fname = os.path.join('pickled_objects', f'X_TRAIN_{target}.txt')
    f = open(fname, 'rb')
    X_TRAIN = pickle.load(f)
    f.close()

    fname = os.path.join('pickled_objects', f'Y_TRAIN_{target}.txt')
    f = open(fname, 'rb')
    Y_TRAIN = pickle.load(f)
    f.close()

    fname = os.path.join('pickled_objects', f'X_VAL_{target}.txt')
    f = open(fname, 'rb')
    X_VAL = pickle.load(f)
    f.close()

    fname = os.path.join('pickled_objects', f'Y_VAL_{target}.txt')
    f = open(fname, 'rb')
    Y_VAL = pickle.load(f)
    f.close()

    # setup model and optimizer for training
    model = create_model(architecture, target, n_layers=layers)
    model.compile(
        optimizer=create_optimizer(optimizer),
        loss='binary_crossentropy',
        metrics=['acc'],
    )

    # configure stuff for tensorboard (for visualization)
    log_dir = f'{model_name}_{architecture}_l{layers}_e{epochs}'
    log_dir = os.path.join('logs', 'fit', log_dir)
    callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
    )

    # start model training
    model.fit(
        x=X_TRAIN,
        y=Y_TRAIN,
        batch_size=16,
        epochs=int(epochs),
        validation_data=(X_VAL, Y_VAL),
        shuffle=True,
        callbacks=[callback],
    )

    # save model weights
    model_weights_dir = os.path.join(
        'saved_models',
        f'{architecture}_weights_{target}',
    )
    current_model_path = os.path.join(model_weights_dir, model_name)
    model.save_weights(current_model_path, overwrite=True)
    print(f'{target}\'s model weights are saved in: {model_weights_dir}')

    if evaluate:
        y_boolmat_val = np.reshape(np.any(X_VAL, axis=2), (-1, 14, 1))
        y_pred, _ = model(X_VAL)
        y_pred = y_pred[y_boolmat_val]
        Y_VAL = Y_VAL[y_boolmat_val]

        print('Confusion Matrix Validation')
        print(confusion_matrix(Y_VAL, np.around(y_pred)))
        print('Validation Accuracy')
        print(accuracy_score(Y_VAL, np.around(y_pred)))
        print('ROC AUC SCORE VAL')
        print(roc_auc_score(Y_VAL, y_pred))
        print('CLASSIFICATION REPORT VAL')
        print(classification_report(Y_VAL, np.around(y_pred)))


def pickle_objects(target='MI', output_dir='pickled_objects', balancing_scheme='duplicate'):
    print(f'[pickle_objects] START: target={target}')

    filenames = [
        f'X_TRAIN_{target}.txt', f'Y_TRAIN_{target}.txt',
        f'X_VAL_{target}.txt', f'Y_VAL_{target}.txt',
        f'X_TEST_{target}.txt', f'Y_TEST_{target}.txt',
        f'features_{target}.txt',
        f'norm_params_{target}.p',
        f'orig_data_{target}.p',
    ]
    output_filenames = [os.path.join(output_dir, name)
                        for name in filenames]

    if all(os.path.isfile(f) for f in output_filenames):
        print(f'[pickle_objects] Pickled files already exist.')
        print(f'[pickle_objects] Will skip target={target}')
        return

    (X_TRAIN, Y_TRAIN, X_VAL, Y_VAL,
     X_TEST, Y_TEST, norm_params, orig_data) = return_data(
        balancing_scheme=balancing_scheme, target=target,
        pad=True, split=True,
    )

    features = return_data(
        return_cols=True, target=target,
        pad=True, split=True,
    )

    output_data = [
        X_TRAIN, Y_TRAIN,
        X_VAL, Y_VAL,
        X_TEST, Y_TEST,
        features,
        norm_params,
        orig_data,
    ]

    # ensure pickled_objects directory exists
    if not os.path.isdir('pickled_objects'):
        os.mkdir('pickled_objects')

    for fname, data in zip(output_filenames, output_data):
        with open(fname, 'wb') as fd:
            pickle.dump(data, fd)

    # output dataset distribution
    dist_fname = os.path.join(output_dir, 'dataset_dist.txt')
    with open(dist_fname, 'a') as fd:
        print('=' * 60, file=fd)
        print(f'Statistics report for target={target}\n', file=fd)

        p_train, n_train = show_stats(Y_TRAIN)
        p_val, n_val = show_stats(Y_VAL)
        p_test, n_test = show_stats(Y_TEST)

        print('[TRAIN]', file=fd)
        print(f'+ {p_train}', file=fd)
        print(f'- {n_train}\n', file=fd)

        print('[VAL]', file=fd)
        print(f'+ {p_val}', file=fd)
        print(f'- {n_val}\n', file=fd)

        print('[TEST]', file=fd)
        print(f'+ {p_test}', file=fd)
        print(f'- {n_test}\n', file=fd)

    print(f'[pickle_objects] DONE: target={target}')


def show_stats(y):
    pos_samples = []
    neg_samples = []

    for i in range(14):
        _reshaped_y = y[:, i, :].reshape(-1)
        total = len(_reshaped_y)
        n_pos = np.sum(_reshaped_y) / total
        n_neg = -np.sum(_reshaped_y - 1) / total

        pos_samples.append(f'{n_pos:.4%}%')
        neg_samples.append(f'{n_neg:.4%}%')

    return pos_samples, neg_samples


def train_models(
    architecture='lstm',
    optimizer='rmsprop',
    layers=1,
    epochs=None,
    evaluate=False,
    balancing_scheme='truncate',
):
    # prepare dataset for MI model
    pickle_objects(target='MI')
    tf.keras.backend.clear_session()

    # prepare dataset for SEPSIS model
    pickle_objects(target='SEPSIS')
    tf.keras.backend.clear_session()

    # prepare dataset for VANCOMYCIN model
    pickle_objects(target='VANCOMYCIN')
    tf.keras.backend.clear_session()

    # train MI model
    train(
        target='MI',
        model_name='model_MI',
        architecture=architecture,
        optimizer=optimizer,
        layers=layers,
        epochs=epochs or 13,
        evaluate=evaluate,
    )
    tf.keras.backend.clear_session()

    # train SEPSIS model
    train(
        target='SEPSIS',
        model_name='model_SEPSIS',
        architecture=architecture,
        optimizer=optimizer,
        layers=layers,
        epochs=epochs or 17,
        evaluate=evaluate,
    )
    tf.keras.backend.clear_session()

    # train VANCOMYCIN model
    train(
        target='VANCOMYCIN',
        model_name='model_VANCOMYCIN',
        architecture=architecture,
        optimizer=optimizer,
        layers=layers,
        epochs=epochs or 14,
        evaluate=evaluate,
    )
    tf.keras.backend.clear_session()


if __name__ == '__main__':
    fire.Fire(train_models)
