import json
import pandas as pd

from sklearn.model_selection import train_test_split

from src.util.util import get_word_start_end_in_sentence

WIC_DATA = 'data/MCL-WiC/'
SUPERGLUE_DATA = 'data/SuperGLUE-WiC/'
COLAB_PREFIX = '/content/MCL-WiC/'

WIC_TRAIN_SUFFIX = 'training/training.en-en.data'
WIC_DEV_SUFFIX = 'dev/multilingual/dev.en-en.data'
WIC_TEST_SUFFIX = 'test/multilingual/test.en-en.data'

SUPERGLUE_TRAIN_SUFFIX = 'train/train.data.txt'
SUPERGLUE_DEV_SUFFIX = 'dev/dev.data.txt'


def read_data_wic(path, read_tags=False):
    with open(path) as f:
        df = pd.DataFrame(json.load(f))

    if read_tags:
        tags_path = path[:path.find('.data')] + '.gold'
        with open(tags_path) as f:
            df = df.merge(pd.DataFrame(json.load(f)))
        df['tag'] = df['tag'].replace({'T': 1, 'F': 0})

    df['lemma'] = df['lemma'].apply(lambda lemma: lemma.lower())
    return df


def read_wic_train(on_colab=True):
    WIC_PREFIX = COLAB_PREFIX + WIC_DATA if on_colab else WIC_DATA
    df = read_data_wic(f'{WIC_PREFIX}{WIC_TRAIN_SUFFIX}', read_tags=True)
    return df


def read_wic_dev(on_colab=True):
    WIC_PREFIX = COLAB_PREFIX + WIC_DATA if on_colab else WIC_DATA
    df = read_data_wic(f'{WIC_PREFIX}{WIC_DEV_SUFFIX}', read_tags=True)
    return df


def read_wic_test(on_colab=True):
    WIC_PREFIX = COLAB_PREFIX + WIC_DATA if on_colab else WIC_DATA
    df = read_data_wic(f'{WIC_PREFIX}{WIC_TEST_SUFFIX}', read_tags=True)
    return df


def read_data_superglue(path, read_tags=True):
    df = pd.read_csv(path, sep='\t', names=['lemma', 'pos', 'word_indices', 'sentence1', 'sentence2'])
    if read_tags:
        tags_path = path[:path.find('.data.txt')] + '.gold.txt'
        df = df.join(pd.read_csv(tags_path, names=['tag']))
        df['tag'] = df['tag'].replace({'T': 1, 'F': 0})

    df['pos'] = df['pos'].replace({'N': 'NOUN', 'V': 'VERB'})

    id_string = path[path.rfind('/') + 1: path.find('.data')]
    df['id'] = df.index
    df['id'] = df['id'].apply(lambda id: id_string + '_superglue.en-en.' + str(id))

    df['start1'] = df.apply(lambda row: get_word_start_end_in_sentence(row)[0][0], axis=1)
    df['end1'] = df.apply(lambda row: get_word_start_end_in_sentence(row)[0][1], axis=1)
    df['start2'] = df.apply(lambda row: get_word_start_end_in_sentence(row)[1][0], axis=1)
    df['end2'] = df.apply(lambda row: get_word_start_end_in_sentence(row)[1][1], axis=1)
    df = df.drop(columns='word_indices')

    return df


def read_superglue_train(on_colab=True):
    SUPERGLUE_PREFIX = COLAB_PREFIX + SUPERGLUE_DATA if on_colab else SUPERGLUE_DATA
    df = read_data_superglue(f'{SUPERGLUE_PREFIX}{SUPERGLUE_TRAIN_SUFFIX}', read_tags=True)
    return df


def read_superglue_dev(on_colab=True):
    SUPERGLUE_PREFIX = COLAB_PREFIX + SUPERGLUE_DATA if on_colab else SUPERGLUE_DATA
    df = read_data_superglue(f'{SUPERGLUE_PREFIX}{SUPERGLUE_DEV_SUFFIX}', read_tags=True)
    return df


def lemma_train_test_split(df, test_size=0.025):
    unique_lemmas = sorted(set(df['lemma'].tolist()))
    train_lemmas, test_lemmas = train_test_split(unique_lemmas, test_size=test_size, random_state=1)
    df_train = df[df['lemma'].isin(train_lemmas)]
    df_test = df[df['lemma'].isin(test_lemmas)]
    return df_train, df_test


def get_train_val_dfs_to_include(only_wic, on_colab=True):
    WIC_PREFIX = COLAB_PREFIX + WIC_DATA if on_colab else WIC_DATA

    df_train_wic = read_wic_train(on_colab)
    df_dev_wic = read_wic_dev(on_colab)

    dfs_to_include = [df_train_wic, df_dev_wic]
    if only_wic:
        return dfs_to_include

    SUPERGLUE_PREFIX = COLAB_PREFIX + SUPERGLUE_DATA if on_colab else SUPERGLUE_DATA

    df_train_superglue = read_superglue_train(on_colab)
    df_dev_superglue = read_superglue_dev(on_colab)

    dfs_to_include.append(df_train_superglue)
    dfs_to_include.append(df_dev_superglue)

    return dfs_to_include


def get_train_val_test_df(use_default_datasets, only_wic=False, on_colab=True):
    WIC_PREFIX = COLAB_PREFIX + WIC_DATA if on_colab else WIC_DATA
    df_test = read_wic_test(on_colab)

    if use_default_datasets:
        df_train = read_wic_train(on_colab)
        df_val = read_wic_dev(on_colab)
        return df_train, df_val, df_test

    global_train_val_df = pd.concat(get_train_val_dfs_to_include(only_wic=only_wic, on_colab=on_colab),
                                    ignore_index=True)
    df_train, df_val = lemma_train_test_split(global_train_val_df)

    return df_train, df_val, df_test
