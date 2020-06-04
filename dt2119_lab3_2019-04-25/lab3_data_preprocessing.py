import os
import numpy as np
from lab3_tools import loadAudio, path2info
from lab3_proto import words2phones, forcedAlignment
from prondict import prondict
from lab1_proto import mfcc, mspec

def feature_extraction_and_force_alignment(filepath, nstates, phoneHMMs):
   """
   handle one .wav file
   """
   samples, samplingrate = loadAudio(filepath)
   wordTrans = list(path2info(filepath)[2])
   phoneTrans = words2phones(wordTrans, prondict)
   stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans
            for stateid in range(nstates[phone])]
   lmfcc_result = mfcc(samples)
   mspec_result = mspec(samples)
   targets = []

   _, viterbi_path = forcedAlignment(lmfcc_result, phoneHMMs, phoneTrans)
   targets = [stateTrans[idx] for idx in viterbi_path.astype(np.int16)] 
   
   return lmfcc_result, mspec_result, targets

def preprocessing(path, nstates, phoneHMMs):
   data = []
   for root, dirs, files in os.walk(path):
      for f in files:
         if f.endswith('.wav'):
            filename = os.path.join(root, f)
            lmfcc, mspec_, targets = feature_extraction_and_force_alignment(filename, nstates, phoneHMMs)
            data.append({'filename': filename, 'lmfcc': lmfcc,
                              'mspec': mspec_, 'targets': targets})
   return data

def precompute_fe_and_fa():
   # run in linux
   phoneHMMs = np.load('lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()
   phones = sorted(phoneHMMs.keys())
   nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
   stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]
   np.save('nstate.npy', nstates)
   np.save('stateList.npy', stateList)

   base_path = '../data/tidigits/'
   train_path = os.path.join(base_path, 'disc_4.1.1', 'tidigits', 'train')
   train_set = preprocessing(train_path, nstates, phoneHMMs)
   #print(train_path)
   np.savez('traindata.npz', train_set)

   test_path = os.path.join(base_path, 'disc_4.2.1', 'tidigits', 'test')
   test_set = preprocessing(test_path, nstates, phoneHMMs)
   np.savez('testdata.npz', test_set)
   
precompute_fe_and_fa()