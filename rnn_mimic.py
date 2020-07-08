''' Recurrent Neural Network in Keras for use on the MIMIC-III '''

import gc
from time import time
import os
import math
import pickle

import numpy as np
import pandas as pd
from pad_sequences import PadSequences
from attention_function import attention_3d_block as Attention

from keras import backend as K
from keras.models import Model, Input, load_model
from keras.layers import Masking, Flatten, Embedding, Dense, LSTM, TimeDistributed
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras import optimizers

from tf_model import Mimic3Lstm
import tensorflow as tf

# from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.join('mimic_database', 'mapped_elements')
PREPROCESSED_FILE = os.path.join(ROOT, 'CHARTEVENTS_preprocessed.csv')


# feature IDs useful for post processing
ID_AGE = {
    'MI': 145,
    'SEPSIS': 149,
    'VANCOMYCIN': 149,
}
ID_MEDS_START = {
    'MI': 147,
    'SEPSIS': 151,
    'VANCOMYCIN': 151,
}


######################################
## MAIN ###
######################################


def get_synth_sequence(n_timesteps=14):
    """

    Returns a single synthetic data sequence of dim (bs,ts,feats)

    Args:
    ----
      n_timesteps: int, number of timesteps to build model for

    Returns:
    -------
      X: npa, numpy array of features of shape (1,n_timesteps,2)
      y: npa, numpy array of labels of shape (1,n_timesteps,1)

    """

    X = np.array([[np.random.rand() for _ in range(n_timesteps)],
                  [np.random.rand() for _ in range(n_timesteps)]])
    X = X.reshape(1, n_timesteps, 2)
    y = np.array([0 if x.sum() < 0.5 else 1 for x in X[0]])
    y = y.reshape(1, n_timesteps, 1)
    return X, y


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


def return_data(synth_data=False, balancer=True, target='MI',
                return_cols=False, tt_split=0.7, val_percentage=0.8,
                cross_val=False, mask=False, dataframe=False,
                time_steps=14, split=True, pad=True):
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
        time_steps : 14 by default, required for padding
        split : creates test train splits
        pad : by default is True, will pad to the time_step value
    Returns:
    -------
        Training and validation splits as well as the number of columns for use in RNN  

    """

    if synth_data:
        no_feature_cols = 2
        X_train = []
        y_train = []

        for i in range(10000):
            X, y = get_synth_sequence(n_timesteps=14)
            X_train.append(X)
            y_train.append(y)
        X_TRAIN = np.vstack(X_train)
        Y_TRAIN = np.vstack(y_train)

    else:
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
            df = PadSequences().pad(df, 1, time_steps, pad_value=pad_value)
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
            int(MATRIX.shape[0]/time_steps), time_steps, MATRIX.shape[1])

        # add my own preprocessing steps
        MATRIX = preprocess_matrix(MATRIX, target)

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

        X_VAL = X_MATRIX[int(tt_split*X_MATRIX.shape[0])
                             :int(val_percentage*X_MATRIX.shape[0])]
        Y_VAL = Y_MATRIX[int(tt_split*Y_MATRIX.shape[0])
                             :int(val_percentage*Y_MATRIX.shape[0])]
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

        if balancer:
            TRAIN = np.concatenate([X_TRAIN, Y_TRAIN], axis=2)
            print(np.where((TRAIN[:, :, -1] == 1).any(axis=1))[0])
            pos_ind = np.unique(
                np.where((TRAIN[:, :, -1] == 1).any(axis=1))[0])
            print(pos_ind)
            np.random.shuffle(pos_ind)
            neg_ind = np.unique(
                np.where(~(TRAIN[:, :, -1] == 1).any(axis=1))[0])
            print(neg_ind)
            np.random.shuffle(neg_ind)
            length = min(pos_ind.shape[0], neg_ind.shape[0])
            total_ind = np.hstack([pos_ind[0:length], neg_ind[0:length]])
            np.random.shuffle(total_ind)
            ind = total_ind
            if target == 'MI':
                ind = pos_ind
            else:
                ind = total_ind
            X_TRAIN = TRAIN[ind, :, 0:-1]
            Y_TRAIN = TRAIN[ind, :, -1]
            Y_TRAIN = Y_TRAIN.reshape(Y_TRAIN.shape[0], Y_TRAIN.shape[1], 1)

    no_feature_cols = X_TRAIN.shape[2]

    if mask:
        print('MASK ACTIVATED')
        X_TRAIN = np.concatenate(
            [np.zeros((X_TRAIN.shape[0], 1, X_TRAIN.shape[2])), X_TRAIN[:, 1::, ::]], axis=1)
        X_VAL = np.concatenate(
            [np.zeros((X_VAL.shape[0], 1, X_VAL.shape[2])), X_VAL[:, 1::, ::]], axis=1)

    if cross_val:
        return (MATRIX, no_feature_cols)

    if split == True:
        return (X_TRAIN, X_VAL, Y_TRAIN, Y_VAL, no_feature_cols,
                X_TEST, Y_TEST, x_test_boolmat, y_test_boolmat,
                x_val_boolmat, y_val_boolmat, means, stds, ORIG_VAL, ORIG_TEST)

    elif split == False:
        return (np.concatenate((X_TRAIN, X_VAL), axis=0),
                np.concatenate((Y_TRAIN, Y_VAL), axis=0), no_feature_cols)


def preprocess_matrix(matrix, target):
    # zero out padding days which have zero values anywhere except "meds" group
    _meds_start = ID_MEDS_START[target]
    _mask = ~matrix[:, :, :_meds_start].any(axis=2)
    matrix[_mask] = 0

    # remove samples with age > 90
    _id_age = ID_AGE[target]
    _temp = matrix.transpose(0, 2, 1)[:, _id_age, :]
    matrix = _temp[~(_temp.max(axis=1) > 90)]

    return matrix


def build_model(no_feature_cols=None, time_steps=7, output_summary=False):
    """

    Assembles RNN with input from return_data function

    Args:
    ----
    no_feature_cols : The number of features being used AKA matrix rank
    time_steps : The number of days in a time block
    output_summary : Defaults to False on returning model summary

    Returns:
    ------- 
    Keras model object

    """
    print(f'time_steps:{time_steps}|no_feature_cols:{no_feature_cols}')

    # input_layer = Input(shape=(time_steps, no_feature_cols))
    # x = Attention(input_layer, time_steps)
    # x = Masking(mask_value=0, input_shape=(time_steps, no_feature_cols))(x)
    # x = LSTM(256, return_sequences=True)(x)
    # preds = TimeDistributed(Dense(1, activation="sigmoid"))(x)
    # model = Model(inputs=input_layer, outputs=preds)

    # RMS = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    # model.compile(optimizer=RMS, loss='binary_crossentropy', metrics=['acc'])

    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=0.001,
        rho=0.0,
        epsilon=1e-08,
    )

    model = Mimic3Lstm(no_feature_cols, time_steps)
    model.compile(
        optimizer=optimizer,  # tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['acc'],
    )

    if output_summary:
        model.model().summary()

    return model


def train(model_name="kaji_mach_0", synth_data=False, target='MI',
          balancer=True, predict=False, return_model=False,
          n_percentage=1.0, time_steps=14, epochs=10):
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

    fname = os.path.join('pickled_objects', f'x_boolmat_val_{target}.txt')
    f = open(fname, 'rb')
    X_BOOLMAT_VAL = pickle.load(f)
    f.close()

    fname = os.path.join('pickled_objects', f'y_boolmat_val_{target}.txt')
    f = open(fname, 'rb')
    Y_BOOLMAT_VAL = pickle.load(f)
    f.close()

    fname = os.path.join('pickled_objects', f'no_feature_cols_{target}.txt')
    f = open(fname, 'rb')
    no_feature_cols = pickle.load(f)
    f.close()

    X_TRAIN = X_TRAIN[0:int(n_percentage*X_TRAIN.shape[0])]
    Y_TRAIN = Y_TRAIN[0:int(n_percentage*Y_TRAIN.shape[0])]

    # build model
    model = build_model(no_feature_cols=no_feature_cols, output_summary=True,
                        time_steps=time_steps)
    print(f'[train] Model successfully built')

    # init callbacks
    log_dir = os.path.join('logs', f'{model_name}_{time()}.log')
    tb_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_grads=False,
        write_images=True,
        write_graph=True,
    )

    # Make checkpoint dir and init checkpointer
    # checkpoint_dir = "./saved_models/{0}".format(model_name)
    # print(f'[train] Checkpoint directory: {checkpoint_dir}')

    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)

    # checkpointer = ModelCheckpoint(
    #     filepath=checkpoint_dir+"/model.{epoch:02d}-{val_loss:.2f}.hdf5",
    #     monitor='val_loss',
    #     verbose=0,
    #     save_best_only=True,
    #     save_weights_only=False,
    #     mode='auto',
    #     period=1
    # )

    # fit
    print(f'[train] Training model...')
    model.fit(
        x=X_TRAIN,
        y=Y_TRAIN,
        batch_size=16,
        epochs=epochs,
        callbacks=[tb_callback],  # , checkpointer],
        validation_data=(X_VAL, Y_VAL),
        shuffle=True,
    )

    # ensure saved_models directory exists
    if not os.path.isdir('saved_models'):
        os.mkdir('saved_models')

    # saved_model_dir = os.path.join('saved_models', f'{model_name}.h5')
    # model.save(saved_model_dir)
    saved_model_dir = os.path.join('saved_models', model_name)
    tf.saved_model.save(model, saved_model_dir)
    print(f'[train] Trained model saved in: {saved_model_dir}')

    if predict:
        print('TARGET: {0}'.format(target))
        Y_PRED, _ = model.predict(X_VAL)
        Y_PRED = Y_PRED[~Y_BOOLMAT_VAL]
        np.unique(Y_PRED)
        Y_VAL = Y_VAL[~Y_BOOLMAT_VAL]
        print('Confusion Matrix Validation')
        print(confusion_matrix(Y_VAL, np.around(Y_PRED)))
        print('Validation Accuracy')
        print(accuracy_score(Y_VAL, np.around(Y_PRED)))
        print('ROC AUC SCORE VAL')
        print(roc_auc_score(Y_VAL, Y_PRED))
        print('CLASSIFICATION REPORT VAL')
        print(classification_report(Y_VAL, np.around(Y_PRED)))

    if return_model:
        return model


def return_loaded_model(model_name="kaji_mach_0"):
    model_dir = os.path.join('saved_models', f'{model_name}.h5')
    loaded_model = load_model(model_dir)

    return loaded_model


def pickle_objects(target='MI', time_steps=14, output_dir='pickled_objects'):
    print(f'[pickle_objects] START: target={target}')

    filenames = [
        f'X_TRAIN_{target}.txt', f'X_VAL_{target}.txt',
        f'Y_TRAIN_{target}.txt', f'Y_VAL_{target}.txt',
        f'X_TEST_{target}.txt', f'Y_TEST_{target}.txt',
        f'x_boolmat_test_{target}.txt', f'y_boolmat_test_{target}.txt',
        f'x_boolmat_val_{target}.txt', f'y_boolmat_val_{target}.txt',
        f'no_feature_cols_{target}.txt', f'features_{target}.txt',
        f'norm_params_{target}.p', f'unnormalized_val_test_{target}.p',
    ]
    output_filenames = [os.path.join(output_dir, name)
                        for name in filenames]

    if all(os.path.isfile(f) for f in output_filenames):
        print(f'[pickle_objects] Pickled files already exist.')
        print(f'[pickle_objects] Will skip target={target}')
        return

    (X_TRAIN, X_VAL, Y_TRAIN, Y_VAL, no_feature_cols,
     X_TEST, Y_TEST, x_boolmat_test, y_boolmat_test,
     x_boolmat_val, y_boolmat_val, means, stds, ORIG_VAL, ORIG_TEST) = return_data(
        balancer=True, target=target, pad=True,
        split=True, time_steps=time_steps
    )

    features = return_data(
        return_cols=True, synth_data=False, target=target,
        pad=True, split=True, time_steps=time_steps
    )

    output_data = [
        X_TRAIN, X_VAL,
        Y_TRAIN, Y_VAL,
        X_TEST, Y_TEST,
        x_boolmat_test, y_boolmat_test,
        x_boolmat_val, y_boolmat_val,
        no_feature_cols, features,
        {'means': means, 'stds': stds},
        {'val': ORIG_VAL, 'test': ORIG_TEST},
    ]

    # ensure pickled_objects directory exists
    if not os.path.isdir('pickled_objects'):
        os.mkdir('pickled_objects')

    for fname, data in zip(output_filenames, output_data):
        with open(fname, 'wb') as fd:
            pickle.dump(data, fd)

    print(f'[pickle_objects] DONE: target={target}')


if __name__ == "__main__":

    pickle_objects(target='MI', time_steps=14)
    K.clear_session()
    pickle_objects(target='SEPSIS', time_steps=14)
    K.clear_session()
    pickle_objects(target='VANCOMYCIN', time_steps=14)

## BIG THREE ##

    K.clear_session()
    train(model_name='kaji_mach_final_no_mask_MI_pad14', epochs=13,
          synth_data=False, predict=True, target='MI', time_steps=14)

    K.clear_session()

    train(model_name='kaji_mach_final_no_mask_VANCOMYCIN_pad14', epochs=14,
          synth_data=False, predict=True, target='VANCOMYCIN', time_steps=14)

    K.clear_session()

    train(model_name='kaji_mach_final_no_mask_SEPSIS_pad14', epochs=17,
          synth_data=False, predict=True, target='SEPSIS', time_steps=14)

    exit()

## REDUCE SAMPLE SIZES ##

## MI ##

    train(model_name='kaji_mach_final_no_mask_MI_pad14_80_percent', epochs=13,
          synth_data=False, predict=True, target='MI', time_steps=14,
          n_percentage=0.80)

    K.clear_session()

    train(model_name='kaji_mach_final_no_mask_MI_pad14_60_percent', epochs=13,
          synth_data=False, predict=True, target='MI', time_steps=14,
          n_percentage=0.60)

    K.clear_session()

    train(model_name='kaji_mach_final_no_mask_MI_pad14_40_percent', epochs=13,
          synth_data=False, predict=True, target='MI', time_steps=14,
          n_percentage=0.40)

    K.clear_session()

    train(model_name='kaji_mach_final_no_mask_MI_pad14_20_percent', epochs=13,
          synth_data=False, predict=True, target='MI', time_steps=14,
          n_percentage=0.20)

    K.clear_session()

    train(model_name='kaji_mach_final_no_mask_MI_pad14_10_percent', epochs=13,
          synth_data=False, predict=True, target='MI', time_steps=14,
          n_percentage=0.10)

    K.clear_session()

    train(model_name='kaji_mach_final_no_mask_MI_pad14_5_percent', epochs=13,
          synth_data=False, predict=True, target='MI', time_steps=14,
          n_percentage=0.05)

    K.clear_session()

# SEPSIS ##

    train(model_name='kaji_mach_final_no_mask_VANCOMYCIN_pad14_80_percent',
          epochs=14, synth_data=False, predict=True, target='VANCOMYCIN',
          time_steps=14, n_percentage=0.80)

    K.clear_session()

    train(model_name='kaji_mach_final_no_mask_VANCOMYCIN_pad14_60_percent',
          epochs=14, synth_data=False, predict=True, target='VANCOMYCIN',
          time_steps=14, n_percentage=0.60)

    K.clear_session()

    train(model_name='kaji_mach_final_no_mask_VANCOMYCIN_pad14_40_percent',
          epochs=14, synth_data=False, predict=True, target='VANCOMYCIN',
          time_steps=14, n_percentage=0.40)

    K.clear_session()

    train(model_name='kaji_mach_final_no_mask_VANCOMYCIN_pad14_20_percent', epochs=14,
          synth_data=False, predict=True, target='VANCOMYCIN', time_steps=14,
          n_percentage=0.20)

    K.clear_session()

    train(model_name='kaji_mach_final_no_mask_VANCOMYCIN_pad14_10_percent',
          epochs=13, synth_data=False, predict=True, target='VANCOMYCIN',
          time_steps=14, n_percentage=0.10)

    K.clear_session()

    train(model_name='kaji_mach_final_no_mask_VANCOMYCIN_pad14_5_percent',
          epochs=13, synth_data=False, predict=True, target='VANCOMYCIN',
          time_steps=14, n_percentage=0.05)

# VANCOMYCIN ##

    train(model_name='kaji_mach_final_no_mask_SEPSIS_pad14_80_percent',
          epochs=17, synth_data=False, predict=True, target='SEPSIS',
          time_steps=14, n_percentage=0.80)

    K.clear_session()

    train(model_name='kaji_mach_final_no_mask_SEPSIS_pad14_60_percent',
          epochs=17, synth_data=False, predict=True, target='SEPSIS',
          time_steps=14, n_percentage=0.60)

    K.clear_session()

    train(model_name='kaji_mach_final_no_mask_SEPSIS_pad14_40_percent',
          epochs=17, synth_data=False, predict=True, target='SEPSIS',
          time_steps=14, n_percentage=0.40)

    K.clear_session()

    train(model_name='kaji_mach_final_no_mask_SEPSIS_pad14_20_percent',
          epochs=17, synth_data=False, predict=True, target='SEPSIS',
          time_steps=14, n_percentage=0.20)

    K.clear_session()

    train(model_name='kaji_mach_final_no_mask_SEPSIS_pad14_10_percent',
          epochs=13, synth_data=False, predict=True, target='SEPSIS',
          time_steps=14, n_percentage=0.10)

    K.clear_session()

    train(model_name='kaji_mach_final_no_mask_SEPSIS_pad14_5_percent',
          epochs=13, synth_data=False, predict=True, target='SEPSIS',
          time_steps=14, n_percentage=0.05)
