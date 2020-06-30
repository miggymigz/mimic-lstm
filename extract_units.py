from collections import defaultdict

import os
import pandas as pd
import pickle

ROOT = 'mimic_database'
RAW_FILE = os.path.join(ROOT, 'CHARTEVENTS.csv')

results = defaultdict(set)
for chunk in pd.read_csv(RAW_FILE, iterator=True, chunksize=10_000_000):
    x = chunk[['ITEMID', 'VALUEUOM']].groupby(
        'ITEMID')['VALUEUOM'].apply(list)
    for k in x.keys():
        values = set(x[k])
        results[k].update(values)

        if len(results[k]) != 1:
            raise AssertionError(f'Units for ID="{k}" has many: {results[k]}')

output_file = os.path.join('pickled_objects', 'feature_units.p')
with open(output_file, 'wb') as fd:
    pickle.dump(results, fd)
