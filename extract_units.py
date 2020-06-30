from collections import defaultdict

import fire
import numpy as np
import os
import pandas as pd
import pickle

ROOT = 'mimic_database'
RAW_FILE = os.path.join(ROOT, 'CHARTEVENTS.csv')


def extract_units(chunksize=10_000_000, output_file='feature_units.p'):
    print(f'INFO - Processing csv with chunksize={chunksize}')

    nan_set = set([np.nan])
    results = defaultdict(set)
    iterator = pd.read_csv(
        RAW_FILE,
        iterator=True,
        chunksize=chunksize,
        low_memory=False
    )

    for i, chunk in enumerate(iterator):
        print(f'INFO - Processing chunk#{i}')

        # item ID is the feature ID
        # valueuom is the unit used by the feature
        columns = ['ITEMID', 'VALUEUOM']

        # group df by item ID
        # list all of the possible units of each item ID
        print('INFO - Grouping items with same ITEMID...')
        group = chunk[columns].groupby('ITEMID')
        agg = group['VALUEUOM'].apply(list)

        print('INFO - Updating results from groups...')
        for key in agg.keys():
            values = set(agg[key])
            results[key].update(values)
            results[key] = results[key].difference(nan_set)

    # ensure output dir exists
    if not os.path.isdir('picked_objects'):
        os.mkdir('pickled_objects')

    # pickle results for later use
    print('INFO - Writing output file...')
    output_file = os.path.join('pickled_objects', output_file)
    with open(output_file, 'wb') as fd:
        pickle.dump(results, fd)


if __name__ == '__main__':
    fire.Fire(extract_units)
