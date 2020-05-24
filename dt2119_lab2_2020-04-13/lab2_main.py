import numpy as np
import matplotlib.pyplot as plt
from lab2_proto import concatHMMs, forward, viterbi, backward
from lab2_tools import log_multivariate_normal_density_diag, logsumexp
from prondict import *
data = np.load('lab2_data.npz', allow_pickle=True)['data']

phoneHMMsAll = np.load('lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()
phoneHMMsOne = np.load('lab2_models_onespkr.npz', allow_pickle=True)['phoneHMMs'].item()

example = np.load('lab2_example.npz', allow_pickle=True)['example'].item()

def plot_p_color_mesh(d2Matrix, caption):
    fig = plt.figure(figsize=(12,6))
    ax = plt.subplot(121)
    ax.set_title(caption)
    plt.pcolormesh(d2Matrix)
    plt.colorbar()
    plt.show()

def get_isolated(pron_dict):
    isolated = {}
    for w, w_list in pron_dict.items():
        isolated[w] = ['sil'] + w_list + ['sil']
    return isolated

def verify_concat_hmms():
    wordHMMs = {}
    isolated = get_isolated(prondict)
    wordHMMs['o'] = concatHMMs(phoneHMMsAll, isolated['o'])
    plot_p_color_mesh(wordHMMs['o']['transmat'], 'word o transmat')
#verify_concat_hmms()

def gaussian_emission_prob():
    example_x = example['lmfcc']
    wordHMMs = {}
    isolated = get_isolated(prondict)
    wordHMMs['o'] = concatHMMs(phoneHMMsAll, isolated['o'])
    lpr = log_multivariate_normal_density_diag(
        example_x, wordHMMs['o']['means'], wordHMMs['o']['covars'])
    
    test_o = data[0]
    test_o_lmfcc = test_o['lmfcc']
    lpr_test = log_multivariate_normal_density_diag(
        test_o_lmfcc, wordHMMs['o']['means'], wordHMMs['o']['covars'])
    plot_p_color_mesh(lpr, 'example log likelihood')
    plot_p_color_mesh(lpr_test, 'test log likelihood')
#gaussian_emission_prob()
    
def forward_algorithm():
    wordHMMs = {}
    isolated = get_isolated(prondict)
    plot_p_color_mesh(example['logalpha'], 'example alpha matrix')
    
    # verify implementation
    wordHMMs['o'] = concatHMMs(phoneHMMsAll, isolated['o'])
    log_st_prob = np.log(wordHMMs['o']['startprob'])
    log_transmat = np.log(wordHMMs['o']['transmat'])
    alpha_matrix = forward(example['obsloglik'], log_st_prob, log_transmat)
    plot_p_color_mesh(alpha_matrix, "hmms all output example alpha matrix")
    
    
    # 44 data labels
    keys_list = [x for x in isolated.keys()]
    scores_models_all = np.zeros((len(data),len(isolated)))
    scores_models_onespkr = np.zeros_like(scores_models_all)

    for j in range(len(keys_list)):
        key = keys_list[j]
        hmms = concatHMMs(phoneHMMsAll, isolated[key])
        log_st_prob = np.log(hmms['startprob'])
        log_transmat = np.log(hmms['transmat'])
        for i in range(len(data)):
            lpr_test = log_multivariate_normal_density_diag(
                data[i]['lmfcc'], hmms['means'], hmms['covars'])
            alpha = forward(lpr_test, log_st_prob, log_transmat)
            scores_models_all[i,j] = logsumexp(alpha[len(alpha) - 1])
        
        hmms = concatHMMs(phoneHMMsOne, isolated[key])
        log_st_prob = np.log(hmms['startprob'])
        log_transmat = np.log(hmms['transmat'])
        for i in range(len(data)):
            lpr_test = log_multivariate_normal_density_diag(
                data[i]['lmfcc'], hmms['means'], hmms['covars'])
            alpha = forward(lpr_test, log_st_prob, log_transmat)
            scores_models_onespkr[i,j] = logsumexp(alpha[len(alpha) - 1])
    
    predict_all = np.argmax(scores_models_all, axis=1)
    predict_one = np.argmax(scores_models_onespkr, axis=1)
    
    label_all = [keys_list[x] for x in predict_all]
    label_one = [keys_list[x] for x in predict_one]
    
    true_label = [data[x]['digit'] for x in range(len(data))]
    print(true_label)
    print(label_all)
    print(label_one)
    
#forward_algorithm()
def viterbi_algorithm():
    wordHMMs = {}
    isolated = get_isolated(prondict)
    
    # verify implementation
    wordHMMs['o'] = concatHMMs(phoneHMMsAll, isolated['o'])
    log_st_prob = np.log(wordHMMs['o']['startprob'])
    log_transmat = np.log(wordHMMs['o']['transmat'])
    vloglik, bestPath = viterbi(example['obsloglik'], log_st_prob, log_transmat)
    alpha_matrix = forward(example['obsloglik'], log_st_prob, log_transmat)
    print('vloglik from viterbi():', vloglik)
    print('vloglik from example:', example['vloglik'])
    
    # plot
    fig = plt.figure(figsize=(12,6))
    ax = plt.subplot(121)
    ax.set_title('viterbi path from Viterbi()')
    plt.pcolormesh(alpha_matrix)
    plt.plot(bestPath, np.arange(len(bestPath)),color='red')
    plt.colorbar()
    plt.show()
    
    fig = plt.figure(figsize=(12,6))
    ax = plt.subplot(121)
    ax.set_title('viterbi path from example')
    plt.pcolormesh(example['logalpha'])
    plt.plot(example['vpath'], np.arange(len(bestPath)),color='red')
    plt.colorbar()
    plt.show()
    
    # 44 data labels
    keys_list = [x for x in isolated.keys()]
    scores_models_all = np.zeros((len(data),len(isolated)))
    scores_models_onespkr = np.zeros_like(scores_models_all)

    for j in range(len(keys_list)):
        key = keys_list[j]
        hmms = concatHMMs(phoneHMMsAll, isolated[key])
        log_st_prob = np.log(hmms['startprob'])
        log_transmat = np.log(hmms['transmat'])
        for i in range(len(data)):
            lpr_test = log_multivariate_normal_density_diag(
                data[i]['lmfcc'], hmms['means'], hmms['covars'])
            loglik, path = viterbi(lpr_test, log_st_prob, log_transmat)
            scores_models_all[i,j] = loglik
        
        hmms = concatHMMs(phoneHMMsOne, isolated[key])
        log_st_prob = np.log(hmms['startprob'])
        log_transmat = np.log(hmms['transmat'])
        for i in range(len(data)):
            lpr_test = log_multivariate_normal_density_diag(
                data[i]['lmfcc'], hmms['means'], hmms['covars'])
            loglik, path = viterbi(lpr_test, log_st_prob, log_transmat)
            scores_models_onespkr[i,j] = loglik
    
    predict_all = np.argmax(scores_models_all, axis=1)
    predict_one = np.argmax(scores_models_onespkr, axis=1)
    
    label_all = [keys_list[x] for x in predict_all]
    label_one = [keys_list[x] for x in predict_one]
    
    true_label = [data[x]['digit'] for x in range(len(data))]
    print(true_label)
    print(label_all)
    print(label_one)
#viterbi_algorithm()
def backward_algorithm():
    wordHMMs = {}
    isolated = get_isolated(prondict)
    plot_p_color_mesh(example['logbeta'], 'example beta matrix')
    
    # verify implementation
    wordHMMs['o'] = concatHMMs(phoneHMMsAll, isolated['o'])
    log_st_prob = np.log(wordHMMs['o']['startprob'])
    log_transmat = np.log(wordHMMs['o']['transmat'])
    beta_matrix = backward(example['obsloglik'], log_st_prob, log_transmat)
    plot_p_color_mesh(beta_matrix, "hmms all output example beta matrix")
    
backward_algorithm()
    