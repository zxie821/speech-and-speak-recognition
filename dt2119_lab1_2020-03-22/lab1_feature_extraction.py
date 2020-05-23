import os

import numpy as np
from scipy.spatial.distance import euclidean
from scipy.cluster import hierarchy
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

from lab1_proto import mfcc, mspec,dtw, plot_p_color_mesh
from lab1_tools import *
output_dir = "results/"
data_dict = np.load('lab1_data.npz', allow_pickle=True)['data']

def q5_feature_correlation():
    mfcc_features_list = []
    mspec_features_list = []
    for sample in data_dict:
        mfcc_feature = mfcc(sample["samples"])
        mfcc_features_list.append(mfcc_feature)
        mspec_feature = mspec(sample["samples"])
        mspec_features_list.append(mspec_feature)

    mfcc_features_list = np.vstack(mfcc_features_list)
    mspec_features_list = np.vstack(mspec_features_list)

    mfcc_cor = np.corrcoef(mfcc_features_list, rowvar = False)
    mspec_cor = np.corrcoef(mspec_features_list, rowvar = False)
    plot_p_color_mesh(mfcc_cor, 'MxM Mfcc correlations')
    plot_p_color_mesh(mspec_cor, 'MxM mspec correlations')
    
def q6_speech_segment_GMM():
    mfcc_features = []
    for i in range(data_dict.shape[0]):
        mfcc_features.append(mfcc(data_dict[0]['samples']))
    mfcc_features = np.vstack(mfcc_features)
        
    n_components = [4,8,16,32]
    idx_seven = [16,17,38,39]
    seven_list = data_dict[idx_seven]
    test_mfcc_seven = []
    for i in range(len(idx_seven)):
        test_mfcc_seven.append(mfcc(seven_list[i]['samples']))
    
    for comp in n_components:
        #train GMM model
        gmm = GaussianMixture(n_components = comp, covariance_type='diag')
        gmm.fit(mfcc_features)

        for i in range(len(idx_seven)):
            test_data = test_mfcc_seven[i]
            prob = gmm.predict_proba(test_data) #compute posterior
            plot_p_color_mesh(prob, 'GMM posterior, n_component=%d, seven #%d'%(comp,i))
            
def q7_global_distance():
    global_distances = np.zeros((44,44))
    all_mfcc = []
    for i in range(44):
        all_mfcc.append(mfcc(data_dict[i]['samples']))
        
    for i in range(44):
        for j in range(44):
            if i==j:
                continue
            elif global_distances[j,i]!=0:
                global_distances[i,j] = global_distances[j,i]
            else:
                global_d, _, acc_d, _ = dtw(all_mfcc[i], all_mfcc[j], euclidean)
                global_distances[i, j] = global_d
    plot_p_color_mesh(global_distances, 'global distance matrix')
    np.save('global distance.npy', global_distances)
    
def q7_compare_utterances():
    global_distances = np.load('global distance.npy')
    idx_seven = [16,17,38,39]
    idx_others = [16,1,5,9]
    seven_distance = global_distances[idx_seven, :][:,idx_seven]
    other_distance = global_distances[idx_others,:][:,idx_others]
    
    plot_p_color_mesh(seven_distance, 'distances among utterances of seven')
    plot_p_color_mesh(other_distance, 'distances among different words')
    
    labels = tidigit2labels(data_dict)
    linkage_matrix = hierarchy.linkage(global_distances, method="complete")
    dn = hierarchy.dendrogram(linkage_matrix, labels = labels, 
                              leaf_rotation=90., leaf_font_size=4)
    plt.savefig(os.path.join(output_dir, "global_distances_dendrogram.png"), dpi = 200)

#q5_feature_correlation()
#q6_speech_segment_GMM()
#q7_global_distance()
q7_compare_utterances()