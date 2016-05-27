import numpy
import os

import numpy
import os

from nmt import train

def main(job_id, params):
    print params
    validerr = train(saveto=params['model'][0],
                     reload_=params['reload'][0],
                     dim_word=params['dim_word'][0],
                     dim=params['dim'][0],
                     n_words=params['n-words'][0],
                     n_words_src=params['n-words-src'][0],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0],
                     patience=1000,
                     maxlen=50,
                     batch_size=80,
                     valid_batch_size=80,
                     validFreq=550,
                     dispFreq=250,
                     saveFreq=550,
                     sampleFreq=550,
                     datasets=['/mnt/data/btec/zh-en/training.zh-en.zh',
                               '/mnt/data/btec/zh-en/training.zh-en.en'],
                     valid_datasets=['/mnt/data/btec/zh-en/dev1_2.zh-en.zh',
                                     '/mnt/data/btec/zh-en/dev1_2.zh-en.en'],
                     dictionaries=['/mnt/data/btec/zh-en/training.zh-en.zh.py2.pkl',
                                   '/mnt/data/btec/zh-en/training.zh-en.en.py2.pkl'],
                     use_dropout=params['use-dropout'][0],
                     overwrite=False)
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['model_adam.npz'],
        'dim_word': [512],
        'dim': [512],
        'n-words-src': [3429],  # 7054
        'n-words' : [3094],
        'optimizer': ['adam'],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [True],
        'learning-rate': [0.001],
        'reload': [False]})
