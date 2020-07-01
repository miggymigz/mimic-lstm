from functools import reduce
from process_mimic import ParseItemID
from tqdm import tqdm

import fire
import math
import numpy as np
import os
import pandas as pd
import pickle

ROOT = 'mimic_database'
RAW_FILE = os.path.join(ROOT, 'CHARTEVENTS.csv')
N_ROWS = 330_712_483


def extract_units(chunksize=10_000_000, output_file='feature_units.p'):
    total = math.ceil(N_ROWS / chunksize)
    results = {}
    iterator = pd.read_csv(
        RAW_FILE,
        iterator=True,
        chunksize=chunksize,
        low_memory=False
    )

    pid = ParseItemID()
    pid.build_dictionary()
    feature_ids = pid.dictionary.values()
    feature_ids = reduce(lambda x, y: x.union(y), feature_ids)

    with tqdm(total=total, position=0) as pbar:
        for i, chunk in enumerate(iterator):
            # set tqdm progress description
            pbar.set_description(f'Processing chunk#{i}')

            # item ID is the feature ID
            # valueuom is the unit used by the feature
            columns = ['ITEMID', 'VALUEUOM']

            # group df by item ID
            # list all of the possible units of each item ID
            pbar.set_description(f'Grouping items for chunk#{i}')
            group = chunk[columns].groupby('ITEMID')
            agg = group['VALUEUOM'].apply(list)
            keys = agg.keys()

            for j, key in enumerate(keys):
                pbar.set_description(
                    f'Updating results of chunk#{i}: {j+1}/{len(keys) + 1}')

                if key in feature_ids:
                    values = set(agg[key])

                    if key in results:
                        results[key].update(values)
                    else:
                        results[key] = values

            # update tqdm progress
            pbar.update()

    # ensure output dir exists
    if not os.path.isdir('pickled_objects'):
        os.mkdir('pickled_objects')

    # pickle results for later use
    print('INFO - Writing output file...')
    output_file = os.path.join('pickled_objects', output_file)
    with open(output_file, 'wb') as fd:
        pickle.dump(results, fd)


if __name__ == '__main__':
    fire.Fire(extract_units)
