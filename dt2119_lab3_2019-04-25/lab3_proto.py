import os
import numpy as np
from lab3_tools import *
from lab2_proto import viterbi, concatHMMs
from lab2_tools import log_multivariate_normal_density_diag
from lab1_proto import mfcc, mspec
from prondict import prondict

def words2phones(wordList, pronDict, addSilence=True, addShortPause=True):
   """ word2phones: converts word level to phone level transcription adding silence

   Args:
      wordList: list of word symbols
      pronDict: pronunciation dictionary. The keys correspond to words in wordList
      addSilence: if True, add initial and final silence
      addShortPause: if True, add short pause model "sp" at end of each word
   Output:
      list of phone symbols
   """
   output_phones = []
   if addSilence:
      output_phones.append('sil')
   for word in wordList:
      output_phones += pronDict[word]
      if addShortPause:
         output_phones.append('sp')
   if addSilence:
      output_phones.append('sil')
   return output_phones

def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
   """ forcedAlignmen: aligns a phonetic transcription at the state level

   Args:
      lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
            computed the same way as for the training of phoneHMMs
      phoneHMMs: set of phonetic Gaussian HMM models
      phoneTrans: list of phonetic symbols to be aligned including initial and
                  final silence

   Returns:
      list of strings in the form phoneme_index specifying, for each time step
      the state from phoneHMMs corresponding to the viterbi path.
   """
   utteranceHMM=concatHMMs(phoneHMMs,phoneTrans)
   emmision=log_multivariate_normal_density_diag(lmfcc,utteranceHMM['means'],utteranceHMM['covars'])
   return viterbi(emmision,np.log(utteranceHMM['startprob']),np.log(utteranceHMM['transmat']))

def hmmLoop(hmmmodels, namelist=None):
    """ Combines HMM models in a loop

    Args:
       hmmmodels: list of dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to combine, if None,
                 all the models in hmmmodels are used

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models
       stateMap: map between states in combinedhmm and states in the
                 input models.

    Examples:
       phoneLoop = hmmLoop(phoneHMMs)
       wordLoop = hmmLoop(wordHMMs, ['o', 'z', '1', '2', '3'])
    """

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

phoneHMMs = np.load('lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()
phones = sorted(phoneHMMs.keys())
nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]
np.savez('stateList.npz', stateList)

base_path = '../data/tidigits/'
train_path = os.path.join(base_path, 'disc_4.1.1', 'tidigits', 'train')
train_set = preprocessing(train_path, nstates, phoneHMMs)
#print(train_path)
np.savez('traindata.npz', train_set)

test_path = os.path.join(base_path, 'disc_4.2.1', 'tidigits', 'test')
test_set = preprocessing(test_path, nstates, phoneHMMs)
np.savez('testdata.npz', test_set)