from functools import reduce
from operator import add

import concurrent.futures
import csv
import fire
import math
import numpy as np
import os
import pandas as pd
import pickle
import re

## Utilities ##


def map_dict(elem, dictionary):
    if elem in dictionary:
        return dictionary[elem]
    else:
        return np.nan

## Proper Classes ##


class ParseItemID(object):

    ''' This class builds the dictionaries depending on desired features '''

    def __init__(self, *, dataset_dir):
        self.dataset_dir = dataset_dir
        self.dictionary = {}

        self.feature_names = [
            'RBCs', 'WBCs', 'platelets', 'hemoglobin', 'hemocrit',
            'atypical lymphocytes', 'bands', 'basophils', 'eosinophils', 'neutrophils',
            'lymphocytes', 'monocytes', 'polymorphonuclear leukocytes',
            'temperature (F)', 'heart rate', 'respiratory rate', 'systolic', 'diastolic',
            'pulse oximetry', 'troponin', 'HDL', 'LDL', 'BUN', 'INR', 'PTT', 'PT', 'triglycerides',
            'creatinine', 'glucose', 'sodium', 'potassium', 'chloride', 'bicarbonate',
            'blood culture', 'urine culture', 'surface culture', 'sputum culture', 'wound culture',
            'Inspired O2 Fraction', 'central venous pressure', 'PEEP Set', 'tidal volume', 'anion gap',
            'daily weight', 'tobacco', 'diabetes', 'history of CV events'
        ]

        self.features = [
            '$^RBC(?! waste)', '$.*wbc(?!.*apache)', '$^platelet(?!.*intake)', '$^hemoglobin',
            '$hematocrit(?!.*Apache)', 'Differential-Atyps', 'Differential-Bands', 'Differential-Basos',
            'Differential-Eos', 'Differential-Neuts', 'Differential-Lymphs', 'Differential-Monos',
            'Differential-Polys', 'temperature f', 'heart rate', 'respiratory rate', 'systolic',
            'diastolic', 'oxymetry(?! )', 'troponin', 'HDL', 'LDL', '$^bun(?!.*apache)', 'INR', 'PTT',
            '$^pt\\b(?!.*splint)(?!.*exp)(?!.*leak)(?!.*family)(?!.*eval)(?!.*insp)(?!.*soft)',
            'triglyceride', '$.*creatinine(?!.*apache)', '(?<!boost )glucose(?!.*apache).*',
            '$^sodium(?!.*apache)(?!.*bicarb)(?!.*phos)(?!.*ace)(?!.*chlo)(?!.*citrate)(?!.*bar)(?!.*PO)',
            '$.*(?<!penicillin G )(?<!urine )potassium(?!.*apache)', '^chloride', 'bicarbonate',
            'blood culture', 'urine culture', 'surface culture', 'sputum culture', 'wound culture',
            'Inspired O2 Fraction', '$Central Venous Pressure(?! )', 'PEEP set', 'tidal volume \(set\)',
            'anion gap', 'daily weight', 'tobacco', 'diabetes', 'CV - past'
        ]

        self.patterns = []
        for feature in self.features:
            if '$' in feature:
                self.patterns.append(feature[1::])
            else:
                self.patterns.append('.*{0}.*'.format(feature))

        # store d_items contents
        d_items_path = os.path.join(self.dataset_dir, 'D_ITEMS.csv')
        self.d_items = pd.read_csv(d_items_path)
        self.d_items.columns = map(str.upper, self.d_items.columns)
        self.d_items = self.d_items[['ITEMID', 'LABEL']]
        self.d_items.dropna(how='any', axis=0, inplace=True)

        self.script_features_names = [
            'epoetin', 'warfarin', 'heparin', 'enoxaparin', 'fondaparinux',
            'asprin', 'ketorolac', 'acetominophen', 'insulin', 'glucagon',
            'potassium', 'calcium gluconate', 'fentanyl', 'magensium sulfate',
            'D5W', 'dextrose', 'ranitidine', 'ondansetron', 'pantoprazole',
            'metoclopramide', 'lisinopril', 'captopril', 'statin', 'hydralazine',
            'diltiazem', 'carvedilol', 'metoprolol', 'labetalol', 'atenolol',
            'amiodarone', 'digoxin(?!.*fab)', 'clopidogrel', 'nitroprusside',
            'nitroglycerin', 'vasopressin', 'hydrochlorothiazide', 'furosemide',
            'atropine', 'neostigmine', 'levothyroxine', 'oxycodone', 'hydromorphone',
            'fentanyl citrate', 'tacrolimus', 'prednisone', 'phenylephrine',
            'norepinephrine', 'haloperidol', 'phenytoin', 'trazodone', 'levetiracetam',
            'diazepam', 'clonazepam', 'propofol', 'zolpidem', 'midazolam',
            'albuterol', 'ipratropium', 'diphenhydramine', '0.9% Sodium Chloride',
            'phytonadione', 'metronidazole', 'cefazolin', 'cefepime', 'vancomycin',
            'levofloxacin', 'cipfloxacin', 'fluconazole', 'meropenem', 'ceftriaxone',
            'piperacillin', 'ampicillin-sulbactam', 'nafcillin', 'oxacillin',
            'amoxicillin', 'penicillin', 'SMX-TMP',
        ]

        self.script_features = [
            'epoetin', 'warfarin', 'heparin', 'enoxaparin', 'fondaparinux',
            'aspirin', 'keterolac', 'acetaminophen', 'insulin', 'glucagon',
            'potassium', 'calcium gluconate', 'fentanyl', 'magnesium sulfate',
            'D5W', 'dextrose', 'ranitidine', 'ondansetron', 'pantoprazole',
            'metoclopramide', 'lisinopril', 'captopril', 'statin', 'hydralazine',
            'diltiazem', 'carvedilol', 'metoprolol', 'labetalol', 'atenolol',
            'amiodarone', 'digoxin(?!.*fab)', 'clopidogrel', 'nitroprusside',
            'nitroglycerin', 'vasopressin', 'hydrochlorothiazide', 'furosemide',
            'atropine', 'neostigmine', 'levothyroxine', 'oxycodone', 'hydromorphone',
            'fentanyl citrate', 'tacrolimus', 'prednisone', 'phenylephrine',
            'norepinephrine', 'haloperidol', 'phenytoin', 'trazodone', 'levetiracetam',
            'diazepam', 'clonazepam', 'propofol', 'zolpidem', 'midazolam',
            'albuterol', '^ipratropium', 'diphenhydramine(?!.*%)(?!.*cream)(?!.*/)',
            '^0.9% sodium chloride(?! )', 'phytonadione', 'metronidazole(?!.*%)(?! desensit)',
            'cefazolin(?! )', 'cefepime(?! )', 'vancomycin', 'levofloxacin',
            'cipfloxacin(?!.*ophth)', 'fluconazole(?! desensit)',
            'meropenem(?! )', 'ceftriaxone(?! desensit)', 'piperacillin',
            'ampicillin-sulbactam', 'nafcillin', 'oxacillin', 'amoxicillin',
            'penicillin(?!.*Desen)', 'sulfamethoxazole'
        ]

        self.script_patterns = ['.*' + feature +
                                '.*' for feature in self.script_features]

    def prescriptions_init(self):
        # ensure PRESCRIPTIONS.csv dataset file exists
        prescriptions_fname = 'PRESCRIPTIONS.csv'
        prescriptions_path = os.path.join(
            self.dataset_dir, prescriptions_fname)
        if not os.path.isfile(prescriptions_path):
            raise FileNotFoundError(f'{prescriptions_fname} does not exist!')

        columns = [
            'ROW_ID', 'SUBJECT_ID', 'HADM_ID',
            'DRUG', 'STARTDATE', 'ENDDATE'
        ]

        self.prescriptions = pd.read_csv(prescriptions_path)
        self.prescriptions.columns = map(str.upper, self.prescriptions.columns)
        self.prescriptions = self.prescriptions[columns]
        self.prescriptions.dropna(how='any', axis=0, inplace=True)

    def query_prescriptions(self, feature_name):
        pattern = '.*{0}.*'.format(feature_name)
        condition = self.prescriptions['DRUG'].str.contains(
            pattern, flags=re.IGNORECASE)
        return self.prescriptions['DRUG'].where(condition).dropna().values

    def extractor(self, feature_name, pattern):
        condition = self.d_items['LABEL'].str.contains(
            pattern, flags=re.IGNORECASE)
        dictionary_value = self.d_items['ITEMID'].where(
            condition).dropna().values.astype('int')
        self.dictionary[feature_name] = set(dictionary_value)

    def query(self, feature_name):
        pattern = '.*{0}.*'.format(feature_name)
        print(pattern)
        condition = self.d_items['LABEL'].str.contains(
            pattern, flags=re.IGNORECASE)
        return self.d_items['LABEL'].where(condition).dropna().values

    def query_pattern(self, pattern):
        condition = self.d_items['LABEL'].str.contains(
            pattern, flags=re.IGNORECASE)
        return self.d_items['LABEL'].where(condition).dropna().values

    def build_dictionary(self):
        assert len(self.feature_names) == len(self.features)
        for feature, pattern in zip(self.feature_names, self.patterns):
            self.extractor(feature, pattern)

    def reverse_dictionary(self, dictionary):
        self.rev = {}
        for key, value in dictionary.items():
            for elem in value:
                self.rev[elem] = key


class MimicParser:

    ''' This class structures the MIMIC III and builds features then makes 24 hour windows '''

    def __init__(self, *, dataset_dir, artifacts_dir, redo=False):
        self.name = 'mimic_assembler'

        self.dataset_dir = dataset_dir
        self.artifacts_dir = artifacts_dir
        self.redo = redo

        self.pid = ParseItemID(dataset_dir=dataset_dir)
        self.pid.build_dictionary()
        self.pid.reverse_dictionary(self.pid.dictionary)

    def normalize_tobacco_values(self, df):
        '''
        values of 'tobacco' item is string and will be dropped if it stays string
        so we convert these values into numbers beforehand
        1. collate all unique values and assert we know them
        '''
        tobacco_ids = self.pid.dictionary['tobacco']
        tobacco_mapping = {
            'Current use or use within 1 month of admission': 1,
            'Stopped more than 1 month ago, but less than 1 year ago': 0.75,
            'Former user - stopped more than 1 year ago': 0.5,
            'Never used': 0,
        }

        # assert values as known
        tobacco_values = np.unique(
            df[df['ITEMID'].isin(tobacco_ids)]['VALUE']
        )
        for v in tobacco_values:
            if v not in tobacco_mapping.keys():
                print(f'[ERROR] Unknown tobacco value: {v}')

        # convert strings to numbers
        predicate1 = df['ITEMID'].isin(tobacco_ids)
        for k, v in tobacco_mapping.items():
            predicate2 = df['VALUE'] == k
            df.loc[predicate1 & predicate2, 'VALUE'] = v
            df.loc[predicate1 & predicate2, 'VALUENUM'] = v

    def reduce_total(self):
        """
        This will filter out rows from CHARTEVENTS.csv that are not feauture relevant
        """
        # ensure input csv exists
        input_fname = 'CHARTEVENTS.csv'
        input_path = os.path.join(self.dataset_dir, input_fname)
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f'{input_fname} does not exist!')

        # do nothing if output file already exists
        output_fname = 'CHARTEVENTS_reduced.csv'
        output_path = os.path.join(self.artifacts_dir, output_fname)
        if not self.redo and os.path.isfile(output_path):
            print(f'[reduce_total] {output_fname} already exists.')
            return

        # make a set of all the item IDs that is relevant
        relevant_item_ids = reduce(
            lambda x, y: x.union(y),
            self.pid.dictionary.values(),
        )

        chunksize = 10_000_000
        columns = [
            'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID',
            'ITEMID', 'CHARTTIME', 'VALUE', 'VALUENUM'
        ]

        iterator = pd.read_csv(
            input_path,
            iterator=True,
            chunksize=chunksize,
            low_memory=False,
        )

        for i, df_chunk in enumerate(iterator):
            print(f'[reduce_total] Processing chunk#{i}')

            # ensure column names are uppercased
            df_chunk.columns = map(str.upper, df_chunk.columns)

            # normalize tobacco values: string -> number
            self.normalize_tobacco_values(df_chunk)

            # drop rows that contain irrelevant features
            condition = df_chunk['ITEMID'].isin(relevant_item_ids)
            df = df_chunk[condition].dropna(axis=0, subset=columns)

            if i == 0:
                df.to_csv(
                    output_path,
                    index=False,
                    columns=columns,
                )
            else:
                df.to_csv(
                    output_path,
                    index=False,
                    columns=columns,
                    header=None,
                    mode='a',
                )

        print(f'[reduce_total] output path: {output_path}')

    def map_files(self, shard_number, filename, low_memory=False):
        ''' HADM minimum is 100001 and maximum is 199999. Shards are built off of those. 
            See if can update based on removing rows from previous buckets to accelerate 
            speed (each iteration 10% faster) This may not be necessary of reduce total 
            works well (there are few features)  '''

        buckets = []
        beg = 100001
        end = 199999
        interval = math.ceil((end - beg)/float(shard_number))

        for i in np.arange(shard_number):
            buckets.append(
                set(np.arange(beg+(i*interval), beg+(interval+(interval*i)))))

        if low_memory == False:

            for i in range(len(buckets)):
                for i, chunk in enumerate(pd.read_csv(filename, iterator=True,
                                                      chunksize=10000000)):
                    print(buckets[i])
                    print(chunk['HADM_ID'].isin(buckets[i]))
                    sliced = chunk[chunk['HADM_ID'].astype(
                        'int').isin(buckets[i])]

                    _path = os.path.join(self.artifacts_dir, f'shard_{i}.csv')
                    sliced.to_csv(_path, index=False)

        else:

            for i in range(len(buckets)):
                with open(filename, 'r') as chartevents:
                    chartevents.seek(0)
                    csvreader = csv.reader(chartevents)

                    _path = os.path.join(self.artifacts_dir, f'shard_{i}.csv')
                    with open(_path, 'w') as shard_writer:
                        csvwriter = csv.writer(shard_writer)
                        for row in csvreader:
                            try:
                                if row[1] == "HADM_ID" or int(row[1]) in buckets[i]:
                                    csvwriter.writerow(row)
                            except ValueError as e:
                                print(row)
                                print(e)

    def create_day_blocks(self):
        """
        Uses pandas to take shards and build them out
        """
        # ensure input csv exists
        input_fname = 'CHARTEVENTS_reduced.csv'
        input_path = os.path.join(self.artifacts_dir, input_fname)
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f'{input_fname} does not exist!')

        # do nothing if output file already exists
        output_fname = 'CHARTEVENTS_reduced_24_hour_blocks.csv'
        output_path = os.path.join(self.artifacts_dir, output_fname)
        if not self.redo and os.path.isfile(output_path):
            print(f'[create_day_blocks] {output_fname} already exists.')
            return

        df = pd.read_csv(input_path)
        df['CHARTDAY'] = df['CHARTTIME'].astype(
            'str').str.split(' ').apply(lambda x: x[0])
        df['HADMID_DAY'] = df['HADM_ID'].astype('str') + '_' + df['CHARTDAY']
        df['FEATURES'] = df['ITEMID'].apply(lambda x: self.pid.rev[x])

        # save a mapping of HADMID_DAY -> patient ID
        hadm_dict = dict(zip(df['HADMID_DAY'], df['SUBJECT_ID']))

        df_src = pd.pivot_table(
            df,
            index='HADMID_DAY',
            columns='FEATURES',
            values='VALUENUM',
            fill_value=np.nan,
            dropna=False,
        )

        df_std = pd.pivot_table(
            df,
            index='HADMID_DAY',
            columns='FEATURES',
            values='VALUENUM',
            aggfunc=np.std,
            fill_value=0,
            dropna=False,
        )
        df_std.columns = [f'{i}_std' for i in list(df_src.columns)]

        df_min = pd.pivot_table(
            df,
            index='HADMID_DAY',
            columns='FEATURES',
            values='VALUENUM',
            aggfunc=np.amin,
            fill_value=np.nan,
            dropna=False,
        )
        df_min.columns = [f'{i}_min' for i in list(df_src.columns)]

        df_max = pd.pivot_table(
            df,
            index='HADMID_DAY',
            columns='FEATURES',
            values='VALUENUM',
            aggfunc=np.amax,
            fill_value=np.nan,
            dropna=False,
        )
        df_max.columns = [f'{i}_max' for i in list(df_src.columns)]

        df2 = pd.concat([df_src, df_std, df_min, df_max], axis=1)

        # remove aggregates of tobacco and daily weights
        del df2['tobacco_std']
        del df2['tobacco_min']
        del df2['tobacco_max']
        del df2['daily weight_std']
        del df2['daily weight_min']
        del df2['daily weight_max']

        rel_columns = list(df2.columns)
        rel_columns = [i for i in rel_columns if '_' not in i]

        for col in rel_columns:
            unique = np.unique(df2[col])
            finite_mask = np.isfinite(unique)
            if len(unique[finite_mask]) <= 2:
                print(f'[create_day_blocks] Will delete "{col}" (std,min,max)')
                del df2[col + '_std']
                del df2[col + '_min']
                del df2[col + '_max']

        for column in df2.columns:
            # precompute the column's median value
            column_median = df2[column].median()

            # replace values > 95% quantile with the median value
            condition = df2[column] > df2[column].quantile(.95)
            df2.loc[condition, column] = column_median

            # fill NA values with the median value
            df2[column] = df2[column].fillna(column_median)

        df2['HADMID_DAY'] = df2.index
        df2['INR'] = df2['INR'] + df2['PT']
        df2['INR_std'] = df2['INR_std'] + df2['PT_std']
        df2['INR_min'] = df2['INR_min'] + df2['PT_min']
        df2['INR_max'] = df2['INR_max'] + df2['PT_max']
        del df2['PT']
        del df2['PT_std']
        del df2['PT_min']
        del df2['PT_max']

        # insert subject ID (patient's unique identifier)
        df2['SUBJECT_ID'] = df2['HADMID_DAY'].apply(
            lambda x: map_dict(x, hadm_dict))

        assert not df2.isna().values.any()
        df2.to_csv(output_path, index=False)

        print(f'[create_day_blocks] output path: {output_path}')

    def add_admissions_columns(self):
        ''' Add demographic columns to create_day_blocks '''
        # ensure input csv exists
        input_fname = 'CHARTEVENTS_reduced_24_hour_blocks.csv'
        input_path = os.path.join(self.artifacts_dir, input_fname)
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f'{input_fname} does not exist!')

        # do nothing if output file already exists
        output_fname = 'CHARTEVENTS_reduced_24_hour_blocks_plus_admissions.csv'
        output_path = os.path.join(self.artifacts_dir, output_fname)
        if not self.redo and os.path.isfile(output_path):
            print(f'[add_admissions_columns] {output_fname} already exists.')
            return

        # ensure ADMISSIONS.csv dataset file exists
        admissions_fname = 'ADMISSIONS.csv'
        admissions_path = os.path.join(self.dataset_dir, admissions_fname)
        if not os.path.isfile(admissions_path):
            raise FileNotFoundError(f'{admissions_fname} does not exist!')

        # load input CSVs
        df_shard = pd.read_csv(input_path)
        df = pd.read_csv(admissions_path)
        df.columns = map(str.upper, df.columns)

        # get admission ID
        df_shard['HADM_ID'] = df_shard['HADMID_DAY'].str.split(
            '_').apply(lambda x: x[0])
        df_shard['HADM_ID'] = df_shard['HADM_ID'].astype('int')

        # determine if patient is black or not
        ethn_dict = dict(zip(df['HADM_ID'], df['ETHNICITY']))
        df_shard['ETHNICITY'] = df_shard['HADM_ID'].apply(
            lambda x: map_dict(x, ethn_dict))
        black_condition = df_shard['ETHNICITY'].str.contains(
            '.*black.*', flags=re.IGNORECASE)
        df_shard['BLACK'] = 0
        df_shard.loc[black_condition, 'BLACK'] = 1
        del df_shard['ETHNICITY']

        # add admit time of a patient
        admittime_dict = dict(zip(df['HADM_ID'], df['ADMITTIME']))
        df_shard['ADMITTIME'] = df_shard['HADM_ID'].apply(
            lambda x: map_dict(x, admittime_dict))

        df_shard.to_csv(output_path, index=False)
        print(f'[add_admissions_columns] output path: {output_path}')

    def add_patient_columns(self):
        '''
        Add demographic columns to create_day_blocks
        '''
        # ensure input csv exists
        input_fname = 'CHARTEVENTS_reduced_24_hour_blocks_plus_admissions.csv'
        input_path = os.path.join(self.artifacts_dir, input_fname)
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f'{input_fname} does not exist!')

        # do nothing if output file already exists
        output_fname = 'CHARTEVENTS_reduced_24_hour_blocks_plus_admissions_plus_patients.csv'
        output_path = os.path.join(self.artifacts_dir, output_fname)
        if not self.redo and os.path.isfile(output_path):
            print(f'[add_patient_columns] {output_path} already exists.')
            return

        # ensure PATIENTS.csv dataset file exists
        patients_fname = 'PATIENTS.csv'
        patients_path = os.path.join(self.dataset_dir, patients_fname)
        if not os.path.isfile(patients_path):
            raise FileNotFoundError(f'{patients_fname} does not exist!')

        df = pd.read_csv(patients_path)
        df.columns = map(str.upper, df.columns)
        dob_dict = dict(zip(df['SUBJECT_ID'], df['DOB']))
        gender_dict = dict(zip(df['SUBJECT_ID'], df['GENDER']))

        df_shard = pd.read_csv(input_path)
        df_shard['DOB'] = df_shard['SUBJECT_ID'].apply(
            lambda x: map_dict(x, dob_dict))
        df_shard['YOB'] = df_shard['DOB'].str.split(
            '-').apply(lambda x: x[0]).astype('int')
        df_shard['ADMITYEAR'] = df_shard['ADMITTIME'].str.split(
            '-').apply(lambda x: x[0]).astype('int')

        # compute patient's age
        # Patients who are older than 89 years old at any time in the database
        # have had their date of birth shifted to obscure their age and comply with HIPAA.
        # so we just simply set their age to 90
        df_shard['AGE'] = df_shard['ADMITYEAR'].subtract(df_shard['YOB'])
        df_shard.loc[df_shard['AGE'] > 89, 'AGE'] = 90

        df_shard['GENDER'] = df_shard['SUBJECT_ID'].apply(
            lambda x: map_dict(x, gender_dict))
        gender_dummied = pd.get_dummies(df_shard['GENDER'], drop_first=True)
        gender_dummied.rename(columns={'M': 'Male', 'F': 'Female'})
        COLUMNS = list(df_shard.columns)
        COLUMNS.remove('GENDER')
        df_shard = pd.concat([df_shard[COLUMNS], gender_dummied], axis=1)
        df_shard.to_csv(output_path, index=False)

        print(f'[add_patient_columns] output path: {output_path}')

    def clean_prescriptions(self):
        '''
        Add prescriptions
        '''
        # ensure input csv exists
        input_fname = 'CHARTEVENTS_reduced_24_hour_blocks_plus_admissions_plus_patients.csv'
        input_path = os.path.join(self.artifacts_dir, input_fname)
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f'{input_fname} does not exist!')

        # do nothing if output file already exists
        output_fname = 'PRESCRIPTIONS_reduced.csv'
        output_path = os.path.join(self.artifacts_dir, output_fname)
        if not self.redo and os.path.isfile(output_path):
            print(f'[clean_prescriptions] {output_fname} already exists.')
            return

        self.pid.prescriptions_init()
        self.pid.prescriptions.drop_duplicates(inplace=True)
        self.pid.prescriptions['DRUG_FEATURE'] = np.nan

        # df_file = pd.read_csv(input_path)
        # hadm_id_array = pd.unique(df_file['HADM_ID'])

        for feature, pattern in zip(self.pid.script_features_names, self.pid.script_patterns):
            condition = self.pid.prescriptions['DRUG'].str.contains(
                pattern,
                flags=re.IGNORECASE,
            )
            self.pid.prescriptions.loc[condition, 'DRUG_FEATURE'] = feature

        self.pid.prescriptions.dropna(
            how='any',
            axis=0,
            inplace=True,
            subset=['DRUG_FEATURE']
        )
        self.pid.prescriptions.to_csv(
            output_path,
            index=False,
        )

        print(f'[clean_prescriptions] output path: {output_path}')

    def add_prescriptions(self, file_name, redo=False):
        print('[add_prescriptions] START')

        output_file = file_name[:-4] + '_plus_scripts.csv'
        if not redo and os.path.isfile(output_file):
            print(f'[clean_prescriptions] {output_file} already exists.')
            print('[clean_prescriptions] Will skip this step.')
            return

        df_file = pd.read_csv(file_name)
        with open('./mimic_database/PRESCRIPTIONS_reduced.csv', 'r') as f:
            csvreader = csv.reader(f)
            with open('./mimic_database/PRESCRIPTIONS_reduced_byday.csv', 'w') as g:
                csvwriter = csv.writer(g)
                first_line = csvreader.__next__()
                print(first_line[0:3] + ['CHARTDAY'] + [first_line[6]])
                csvwriter.writerow(
                    first_line[0:3] + ['CHARTDAY'] + [first_line[6]])
                for row in csvreader:
                    for i in pd.date_range(row[3], row[4]).strftime('%Y-%m-%d'):
                        csvwriter.writerow(row[0:3] + [i] + [row[6]])

        df = pd.read_csv('./mimic_database/PRESCRIPTIONS_reduced_byday.csv')
        df['CHARTDAY'] = df['CHARTDAY'].str.split(' ').apply(lambda x: x[0])
        df['HADMID_DAY'] = df['HADM_ID'].astype('str') + '_' + df['CHARTDAY']
        df['VALUE'] = 1

        cols = ['HADMID_DAY', 'DRUG_FEATURE', 'VALUE']
        df = df[cols]

        df_pivot = pd.pivot_table(
            df, index='HADMID_DAY', columns='DRUG_FEATURE', values='VALUE', fill_value=0, aggfunc=np.amax)
        df_pivot.reset_index(inplace=True)

        df_merged = pd.merge(df_file, df_pivot, on='HADMID_DAY', how='outer')

        del df_merged['HADM_ID']
        df_merged['HADM_ID'] = df_merged['HADMID_DAY'].str.split(
            '_').apply(lambda x: x[0])
        df_merged.fillna(0, inplace=True)

        df_merged['dextrose'] = df_merged['dextrose'] + df_merged['D5W']
        del df_merged['D5W']

        df_merged.to_csv(output_file, index=False)

        print(f'[add_prescriptions] output file: {output_file}')

    def add_icd_infect(self, file_name, redo=False):
        print('[add_icd_infect] START')

        output_file = file_name[:-4] + '_plus_icds.csv'
        if not redo and os.path.isfile(output_file):
            print(f'[clean_prescriptions] {output_file} already exists.')
            print('[clean_prescriptions] Will skip this step.')
            return

        df_icd = pd.read_csv('./mimic_database/PROCEDURES_ICD.csv')
        df_micro = pd.read_csv('./mimic_database/MICROBIOLOGYEVENTS.csv')
        self.suspect_hadmid = set(pd.unique(df_micro['HADM_ID']).tolist())
        df_icd_ckd = df_icd[df_icd['ICD9_CODE'] == 585]

        self.ckd = set(df_icd_ckd['HADM_ID'].values.tolist())

        df = pd.read_csv(file_name)
        df['CKD'] = df['HADM_ID'].apply(lambda x: 1 if x in self.ckd else 0)
        df['Infection'] = df['HADM_ID'].apply(
            lambda x: 1 if x in self.suspect_hadmid else 0)
        df.to_csv(output_file, index=False)

        print(f'[add_icd_infect] output file: {output_file}')

    def add_notes(self, file_name, redo=False):
        print('[add_notes] START')

        output_file = file_name[:-4] + '_plus_notes.csv'
        if not redo and os.path.isfile(output_file):
            print(f'[clean_prescriptions] {output_file} already exists.')
            print('[clean_prescriptions] Will skip this step.')
            return

        df = pd.read_csv('./mimic_database/NOTEEVENTS.csv')
        df_rad_notes = df[['TEXT', 'HADM_ID']][df['CATEGORY'] == 'Radiology']
        CTA_bool_array = df_rad_notes['TEXT'].str.contains(
            'CTA', flags=re.IGNORECASE)
        CT_angiogram_bool_array = df_rad_notes['TEXT'].str.contains(
            'CT angiogram', flags=re.IGNORECASE)
        chest_angiogram_bool_array = df_rad_notes['TEXT'].str.contains(
            'chest angiogram', flags=re.IGNORECASE)
        cta_hadm_ids = np.unique(
            df_rad_notes['HADM_ID'][CTA_bool_array].dropna())
        CT_angiogram_hadm_ids = np.unique(
            df_rad_notes['HADM_ID'][CT_angiogram_bool_array].dropna())
        chest_angiogram_hadm_ids = np.unique(
            df_rad_notes['HADM_ID'][chest_angiogram_bool_array].dropna())
        hadm_id_set = set(cta_hadm_ids.tolist())
        hadm_id_set.update(CT_angiogram_hadm_ids)
        print(len(hadm_id_set))
        hadm_id_set.update(chest_angiogram_hadm_ids)
        print(len(hadm_id_set))

        df2 = pd.read_csv(file_name)
        df2['ct_angio'] = df2['HADM_ID'].apply(
            lambda x: 1 if x in hadm_id_set else 0)
        df2.to_csv(output_file, index=False)

        print(f'[add_notes] output file: {output_file}')


def main(root='mimic_database', redo=False):
    # ensure dataset directory exists
    if not os.path.isdir(root):
        raise FileNotFoundError(f'Directory "{root}" does not exist.')

    # all of the artifacts built by this preprocessing
    # will be placed in the `mapped_elements` folder
    # create it if it does not exist
    artifacts_dir = os.path.join(root, 'mapped_elements')
    if not os.path.isdir(artifacts_dir):
        os.mkdir(artifacts_dir)

    # create instance of parser to be used for dataset preprocessing
    mp = MimicParser(
        dataset_dir=root,
        artifacts_dir=artifacts_dir,
        redo=redo,
    )

    # filter out rows from CHARTEVENTS.csv
    # rows that doesn't contain relevant features will be dropped
    mp.reduce_total()

    # reduce rows into days
    # aggregating feature multiple measurements (within the day) into avg/min/max/std
    mp.create_day_blocks()

    # add patient admission information
    # patient ethnicity and admit time
    mp.add_admissions_columns()

    # add patient demographic information
    # patient age and gender
    mp.add_patient_columns()

    # TODO: add comment
    mp.clean_prescriptions()
    # mp.add_prescriptions(fpath, redo=redo)

    # fpath = fpath[:-4] + '_plus_scripts.csv'
    # mp.add_icd_infect(fpath, redo=redo)

    # fpath = fpath[:-4] + '_plus_icds.csv'
    # mp.add_notes(fpath, redo=redo)


if __name__ == '__main__':
    fire.Fire(main)
