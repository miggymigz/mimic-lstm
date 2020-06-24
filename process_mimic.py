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

ROOT = "./mimic_database/"

## Utilities ##


def map_dict(elem, dictionary):
    if elem in dictionary:
        return dictionary[elem]
    else:
        return np.nan

## Proper Classes ##


class ParseItemID(object):

    ''' This class builds the dictionaries depending on desired features '''

    def __init__(self):
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
            if '$' not in feature:
                self.patterns.append('.*{0}.*'.format(feature))
            elif '$' in feature:
                self.patterns.append(feature[1::])

        self.d_items = pd.read_csv(
            ROOT + 'D_ITEMS.csv',
            usecols=['ITEMID', 'LABEL']
        )
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
        self.prescriptions = pd.read_csv(ROOT + 'PRESCRIPTIONS.csv',
                                         usecols=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'DRUG',
                                                  'STARTDATE', 'ENDDATE'])
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


class MimicParser(object):

    ''' This class structures the MIMIC III and builds features then makes 24 hour windows '''

    def __init__(self):
        self.name = 'mimic_assembler'
        self.pid = ParseItemID()
        self.pid.build_dictionary()
        self.features = self.pid.features

    def reduce_total(self, filepath, redo=False):
        ''' This will filter out rows from CHARTEVENTS.csv that are not feauture relevant '''
        print('[reduce_total] START')

        output_file = ROOT + './mapped_elements/CHARTEVENTS_reduced.csv'
        if not redo and os.path.isfile(output_file):
            print(f'[reduce_total] {output_file} already exists.')
            print('[reduce_total] Will skip this step.')
            return

        pid = ParseItemID()
        pid.build_dictionary()
        dict_values = pid.dictionary.values()
        dict_values = reduce(lambda x, y: x.union(y), dict_values)

        chunksize = 10_000_000
        columns = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID',
                   'ITEMID', 'CHARTTIME', 'VALUE', 'VALUENUM']
        iterator = pd.read_csv(
            filepath,
            iterator=True,
            chunksize=chunksize,
        )

        for i, df_chunk in enumerate(iterator):
            print(f'[reduce_total] Processing chunk#{i}')

            df = df_chunk[df_chunk['ITEMID'].isin(dict_values)]
            df = df.dropna(axis=0, subset=columns)

            if i == 0:
                df.to_csv(
                    output_file,
                    index=False,
                    columns=columns,
                )
            else:
                df.to_csv(
                    output_file,
                    index=False,
                    columns=columns,
                    header=None,
                    mode='a',
                )

        print(f'[reduce_total] output file: {output_file}')

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
                    sliced.to_csv(
                        ROOT + 'mapped_elements/shard_{0}.csv'.format(i), index=False)

        else:

            for i in range(len(buckets)):
                with open(filename, 'r') as chartevents:
                    chartevents.seek(0)
                    csvreader = csv.reader(chartevents)
                    with open(ROOT+'mapped_elements/shard_{0}.csv'.format(i), 'w') as shard_writer:
                        csvwriter = csv.writer(shard_writer)
                        for row in csvreader:
                            try:
                                if row[1] == "HADM_ID" or int(row[1]) in buckets[i]:
                                    csvwriter.writerow(row)
                            except ValueError as e:
                                print(row)
                                print(e)

    def create_day_blocks(self, file_name, redo=False):
        ''' Uses pandas to take shards and build them out '''

        print('[create_day_blocks] START')

        output_file = file_name[:-4] + '_24_hour_blocks.csv'
        if not redo and os.path.isfile(output_file):
            print(f'[create_day_blocks] {output_file} already exists.')
            print('[create_day_blocks] Will skip this step.')
            return

        pid = ParseItemID()
        pid.build_dictionary()
        pid.reverse_dictionary(pid.dictionary)

        df = pd.read_csv(file_name)
        df['CHARTDAY'] = df['CHARTTIME'].astype('str') \
            .str.split(' ') \
            .apply(lambda x: x[0])
        df['HADMID_DAY'] = df['HADM_ID'].astype('str') + '_' + df['CHARTDAY']
        df['FEATURES'] = df['ITEMID'].apply(lambda x: pid.rev[x])
        print(f'[DEBUG] df.shape={df.shape}')

        self.hadm_dict = dict(zip(df['HADMID_DAY'], df['SUBJECT_ID']))

        df2 = pd.pivot_table(
            df, index='HADMID_DAY', columns='FEATURES', values='VALUENUM', fill_value=np.nan)

        df3 = pd.pivot_table(df, index='HADMID_DAY', columns='FEATURES',
                             values='VALUENUM', aggfunc=np.std, fill_value=0)
        df3.columns = ["{0}_std".format(i) for i in list(df2.columns)]

        df4 = pd.pivot_table(df, index='HADMID_DAY', columns='FEATURES',
                             values='VALUENUM', aggfunc=np.amin, fill_value=np.nan)
        df4.columns = ["{0}_min".format(i) for i in list(df2.columns)]

        df5 = pd.pivot_table(df, index='HADMID_DAY', columns='FEATURES',
                             values='VALUENUM', aggfunc=np.amax, fill_value=np.nan)

        df2 = pd.concat([df2, df3, df4, df5], axis=1, sort=True)
        if 'tobacco' in df2.columns:
            print('[create_day_blocks] WARN: "tobacco" not in df2.columns')
            df2['tobacco'].apply(lambda x: np.around(x))

        if 'daily weight_std' in df2:
            print('[create_day_blocks] Will delete "daily weight_std" in df2')
            del df2['daily weight_std']
        if 'daily weight_min' in df2:
            print('[create_day_blocks] Will delete "daily weight_min" in df2')
            del df2['daily weight_min']
        if 'daily weight_max' in df2:
            print('[create_day_blocks] Will delete "daily weight_max" in df2')
            del df2['daily weight_max']
        if 'tobacco_std' in df2:
            print('[create_day_blocks] Will delete "tobacco_std" in df2')
            del df2['tobacco_std']
        if 'tobacco_min' in df2:
            print('[create_day_blocks] Will delete "tobacco_min" in df2')
            del df2['tobacco_min']
        if 'tobacco_max' in df2:
            print('[create_day_blocks] Will delete "tobacco_max" in df2')
            del df2['tobacco_max']

        rel_columns = list(df2.columns)
        rel_columns = [i for i in rel_columns if '_' not in i]

        for col in rel_columns:
            if len(np.unique(df2[col])[np.isfinite(np.unique(df2[col]))]) <= 2:
                print(f'[create_day_blocks] Will delete "{col}" (std,min,max)')
                del df2[col + '_std']
                del df2[col + '_min']
                del df2[col + '_max']

        for i in list(df2.columns):
            print(f'[create_day_blocks] Will average column "{i}"')
            df2[i][df2[i] > df2[i].quantile(.95)] = df2[i].median()
            # if i != 'troponin':
            #     df2[i] = df2[i].where(df2[i] > df2[i].quantile(.875)).fillna(df2[i].median())

        for i in list(df2.columns):
            df2[i].fillna(df2[i].median(), inplace=True)

        df2['HADMID_DAY'] = df2.index
        df2['INR'] = df2['INR'] + df2['PT']
        df2['INR_std'] = df2['INR_std'] + df2['PT_std']
        df2['INR_min'] = df2['INR_min'] + df2['PT_min']
        df2['INR_max'] = df2['INR_max'] + df2['PT_max']
        del df2['PT']
        del df2['PT_std']
        del df2['PT_min']
        del df2['PT_max']
        df2.dropna(thresh=int(0.75*len(df2.columns)), axis=0, inplace=True)
        df2.to_csv(output_file, index=False)

        print(f'[create_day_blocks] output file: {output_file}')

    def add_admissions_columns(self, file_name, redo=False):
        ''' Add demographic columns to create_day_blocks '''

        print('[add_admissions_columns] START')

        output_file = file_name[:-4] + '_plus_admissions.csv'
        if not redo and os.path.isfile(output_file):
            print(f'[add_admissions_columns] {output_file} already exists.')
            print('[add_admissions_columns] Will skip this step.')
            return

        df = pd.read_csv('./mimic_database/ADMISSIONS.csv')
        ethn_dict = dict(zip(df['HADM_ID'], df['ETHNICITY']))
        admittime_dict = dict(zip(df['HADM_ID'], df['ADMITTIME']))
        df_shard = pd.read_csv(file_name)
        df_shard['HADM_ID'] = df_shard['HADMID_DAY'].str.split(
            '_').apply(lambda x: x[0])
        df_shard['HADM_ID'] = df_shard['HADM_ID'].astype('int')
        df_shard['ETHNICITY'] = df_shard['HADM_ID'].apply(
            lambda x: map_dict(x, ethn_dict))
        black_condition = df_shard['ETHNICITY'].str.contains(
            '.*black.*', flags=re.IGNORECASE)
        df_shard['BLACK'] = 0
        df_shard['BLACK'][black_condition] = 1
        del df_shard['ETHNICITY']
        df_shard['ADMITTIME'] = df_shard['HADM_ID'].apply(
            lambda x: map_dict(x, admittime_dict))
        df_shard.to_csv(output_file, index=False)

        print(f'[add_admissions_columns] output file: {output_file}')

    def add_patient_columns(self, file_name, redo=False):
        ''' Add demographic columns to create_day_blocks '''

        print('[add_patient_columns] START')

        output_file = file_name[:-4] + '_plus_patients.csv'
        if not redo and os.path.isfile(output_file):
            print(f'[add_patient_columns] {output_file} already exists.')
            print('[add_patient_columns] Will skip this step.')
            return

        df = pd.read_csv('./mimic_database/PATIENTS.csv')
        dob_dict = dict(zip(df['SUBJECT_ID'], df['DOB']))
        gender_dict = dict(zip(df['SUBJECT_ID'], df['GENDER']))
        df_shard = pd.read_csv(file_name)
        df_shard['SUBJECT_ID'] = df_shard['HADMID_DAY'].apply(lambda x:
                                                              map_dict(x, self.hadm_dict))
        df_shard['DOB'] = df_shard['SUBJECT_ID'].apply(
            lambda x: map_dict(x, dob_dict))

        df_shard['YOB'] = df_shard['DOB'].str.split(
            '-').apply(lambda x: x[0]).astype('int')
        df_shard['ADMITYEAR'] = df_shard['ADMITTIME'].str.split(
            '-').apply(lambda x: x[0]).astype('int')

        df_shard['AGE'] = df_shard['ADMITYEAR'].subtract(df_shard['YOB'])
        df_shard['GENDER'] = df_shard['SUBJECT_ID'].apply(
            lambda x: map_dict(x, gender_dict))
        gender_dummied = pd.get_dummies(df_shard['GENDER'], drop_first=True)
        gender_dummied.rename(columns={'M': 'Male', 'F': 'Female'})
        COLUMNS = list(df_shard.columns)
        COLUMNS.remove('GENDER')
        df_shard = pd.concat([df_shard[COLUMNS], gender_dummied], axis=1)
        df_shard.to_csv(output_file, index=False)

        print(f'[add_patient_columns] output file: {output_file}')

    def clean_prescriptions(self, file_name, redo=False):
        ''' Add prescriptions '''

        print('[clean_prescriptions] START')

        output_file = './mimic_database/PRESCRIPTIONS_reduced.csv'
        if not redo and os.path.isfile(output_file):
            print(f'[clean_prescriptions] {output_file} already exists.')
            print('[clean_prescriptions] Will skip this step.')
            return

        pid = ParseItemID()
        pid.prescriptions_init()
        pid.prescriptions.drop_duplicates(inplace=True)
        pid.prescriptions['DRUG_FEATURE'] = np.nan

        df_file = pd.read_csv(file_name)
        hadm_id_array = pd.unique(df_file['HADM_ID'])

        for feature, pattern in zip(pid.script_features_names, pid.script_patterns):
            condition = pid.prescriptions['DRUG'].str.contains(
                pattern, flags=re.IGNORECASE)
            pid.prescriptions['DRUG_FEATURE'][condition] = feature

        pid.prescriptions.dropna(
            how='any', axis=0, inplace=True, subset=['DRUG_FEATURE'])
        pid.prescriptions.to_csv(output_file, index=False)

        print(f'[clean_prescriptions] output file: {output_file}')

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


def main(redo=False):
    mp = MimicParser()
    pid = ParseItemID()
    pid.build_dictionary()

    fpath = ROOT + 'CHARTEVENTS.csv'
    mp.reduce_total(fpath, redo=redo)

    fpath = ROOT + 'mapped_elements/CHARTEVENTS_reduced.csv'
    mp.create_day_blocks(fpath, redo=redo)

    fpath = fpath[:-4] + '_24_hour_blocks.csv'
    mp.add_admissions_columns(fpath, redo=redo)

    fpath = fpath[:-4] + '_plus_admissions.csv'
    mp.add_patient_columns(fpath, redo=redo)

    fpath = fpath[:-4] + '_plus_patients.csv'
    mp.clean_prescriptions(fpath, redo=redo)
    mp.add_prescriptions(fpath, redo=redo)

    fpath = fpath[:-4] + '_plus_scripts.csv'
    mp.add_icd_infect(fpath, redo=redo)

    fpath = fpath[:-4] + '_plus_icds.csv'
    mp.add_notes(fpath, redo=redo)


if __name__ == '__main__':
    fire.Fire(main)
