import numpy as np
from lab3_tools import path2info
def split_train_validation_data():
   traindata = np.load("traindata.npz", allow_pickle=True)['arr_0']
   
   male_list = []
   female_list = []

   record_id = []
   for i in range(len(traindata)):
      f_name = traindata[i]['filename']
      gender, id_num, _, _ = path2info(f_name)
      if id_num not in record_id:
         record_id.append(id_num)
         if gender == 'man':
            male_list.append(id_num)
         else:
            female_list.append(id_num)
   print(len(female_list), "and ", len(male_list))
   train_size = (int)(len(record_id) * 0.9)
   male_ratio = len(male_list) / train_size
   male_train_size = (int)(train_size * male_ratio)
   print(male_train_size, ' ', train_size)
   male_train_ids = np.random.choice(male_list, male_train_size, replace=False)
   female_train_ids = np.random.choice(female_list, train_size-male_train_size,replace=False)
   train_ids = np.concatenate([male_train_ids, female_train_ids])
   
   train_data = []
   vali_data = []
   for i in range(len(traindata)):
      _, id_num, _, _ = path2info(traindata[i]['filename'])
      if id_num in train_ids:
         train_data.append(traindata[i])
      else:
         vali_data.append(traindata[i])
   print('num of male speaker for training:', len(male_train_ids))
   print('num of female speaker for training:', len(female_train_ids))
   print('num of total train data:', len(train_data))
   print('num of total val data:', len(vali_data))
   np.save('train_split_data.npy', train_data)
   np.save('vali_split_data.npy', vali_data)

split_train_validation_data()