import tsl
import pickle
import numpy as np
import torch
import torchtime
from torch.utils.data import DataLoader

# print(f"tsl version  : {tsl.__version__}")
# print(f"torch version: {torch.__version__}")



##### Climate Dataset
from tsl.datasets import ClimateCapitals, MetrLA
# hourly = ClimateCapitals(name='climateHourly', root='./data/NASA_data')
# daily = ClimateCapitals(name='climateDaily', root='./data/NASA_data')



# # UEA/UCR
from tsl.datasets import UEA_archive
from torchtime.data import UEA

# datasets_list = [
#     'ArticularyWordRecognition',
#     'AtrialFibrillation',
#     'BasicMotions',
#     'CharacterTrajectories',
#     'Cricket',
#     ##### 'DuckDuckGeese',
#     'EigenWorms',
#     'Epilepsy',
#     'EthanolConcentration',
#     'ERing',
#     #### 'FaceDetection',
#     'FingerMovements',
#     'HandMovementDirection',
#     'Handwriting',
#     'Heartbeat',
#     #### 'InsectWingbeat',
#     'JapaneseVowels',
#     'Libras',
#     'LSST',
#     'MotorImagery',
#     'NATOPS',
#     'PenDigits',
#     #### 'PEMS-SF',
#     'Phoneme',
#     'RacketSports',
#     'SelfRegulationSCP1',
#     'SelfRegulationSCP2',
#     'SpokenArabicDigits',
#     'StandWalkJump',
#     'UWaveGestureLibrary'
#     ]


# for dataset_name in datasets_list:
#     uea_dataset = UEA(
#         dataset=dataset_name,
#         time=True,
#         mask=False,
#         delta=False,
#         standardise=True,
#         path=f'./data',
#         split='train',
#         train_prop=0.8,  # 70% training
#         seed=1234,
#     )
    
#     print(uea_dataset.X_train.shape)
#     print(uea_dataset.X_val.shape)
#     print('total N = ', uea_dataset.X_train.shape[0] + uea_dataset.X_val.shape[0])
    
    
#     concat_data    = torch.cat([uea_dataset.X_train, uea_dataset.X_val], dim=0).numpy()
#     concat_label   = torch.cat([uea_dataset.y_train, uea_dataset.y_val], dim=0).numpy()
#     concat_length  = torch.cat([uea_dataset.length_train, uea_dataset.length_val], dim=0).numpy()
    
#     file = open(f'./data/UEA/{dataset_name}.npz', 'wb')
#     np.savez(file, concat_data, concat_label, concat_length)
    
#     data = UEA_archive(name=dataset_name, root='./data/UEA')
    
#     print(data)
#     # file = open(f'./data/UEA/{dataset_name}.npz', 'rb')
#     # npzfile = np.load(file)
#     # df = npzfile['arr_0']    
    
#     # hf = h5py.File(f'./data/UEA/{dataset_name}.h5', 'w')
#     # hf.create_dataset('data',    data=concat_data)
#     # hf.create_dataset('labels',  data=concat_label)
#     # hf.create_dataset('lengths', data=concat_length)
    