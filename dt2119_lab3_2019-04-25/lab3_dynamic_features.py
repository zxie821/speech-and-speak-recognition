import numpy as np
from sklearn.preprocessing import StandardScaler
def convert_dynamic_features(data, feature_name):
    result = []
    feature_len = len(data)
    for i in range(feature_len):
        if i>=3 and i<feature_len - 3:
            feature = data[i-3:i+4]
        elif i==0:
            feature = data[[3,2,1,0,1,2,3]]
        elif i==1:
            feature = data[[2,1,0,1,2,3,4]]
        elif i==2:
            feature = data[[1,0,1,2,3,4,5]]
        elif i==feature_len-3:
            feature = data[[i-4, i-3,i-2,i-1,i,i+1,i+2]]
        elif i==feature_len-2:
            feature = data[[i-5, i-4,i-3,i-2,i-1,i,i+1]]
        elif i==feature_len-1:
            feature = data[[i-6, i-5,i-4,i-3,i-2,i-1,i]]
        else:
            feature = None
        result.append(feature.flatten())
    return np.stack(result)

def dynamic_features(dataset):
    for data in dataset:
        data['dylmfcc'] = convert_dynamic_features(data['lmfcc'], 'lmfcc')
        data['dymspec'] = convert_dynamic_features(data['mspec'], 'mspec')
    return dataset
def compute_all_dataset_dy_features():
    test_dataset = np.load('testdata.npz', allow_pickle=True)['arr_0']
    dynamic_features(test_dataset)
    np.save('all_features_test_data.npy', test_dataset)
    
    train_dataset = np.load('train_split_data.npy', allow_pickle=True)
    dynamic_features(train_dataset)
    np.save('all_features_train_data.npy', train_dataset)
    
    vali_dataset = np.load('vali_split_data.npy', allow_pickle=True)
    dynamic_features(vali_dataset)
    np.save('all_features_val_data.npy', vali_dataset)

def normalization(dataset, feature_name, name_prefix, scalar=None):
    # feature_name = ='dylmfcc' or 'dymspec'
    feature_list = []
    label_list = []
    for data_dict in dataset:    
        feature_list.append(data_dict[feature_name])
        label_list+=data_dict['targets']
    features = np.vstack(feature_list).astype('float32')
    if scalar is None:
        scalar = StandardScaler()
        features = scalar.fit_transform(features)
    else:
        features = scalar.transform(features)
    
    np.save(name_prefix+'_norm_feature_'+feature_name+'.npy', features)
    np.save(name_prefix+'_norm_label_labels.npy', label_list)
    return scalar

def normalization_all_datasets():
    dataset = np.load('all_features_train_data.npy', allow_pickle=True)
    scalar_dylmfcc = normalization(dataset, 'dylmfcc', 'train')
    scalar_dymspec = normalization(dataset, 'dymspec', 'train')
    scalar_lmfcc = normalization(dataset, 'lmfcc', 'train')
    scalar_mspec = normalization(dataset, 'mspec', 'train')
    
    dataset = np.load('all_features_val_data.npy', allow_pickle=True)
    normalization(dataset, 'dylmfcc', 'val', scalar_dylmfcc)
    normalization(dataset, 'dymspec', 'val', scalar_dymspec)
    normalization(dataset, 'lmfcc', 'val', scalar_lmfcc)
    normalization(dataset, 'mspec', 'val', scalar_mspec)
    
    dataset = np.load('all_features_test_data.npy', allow_pickle=True)
    normalization(dataset, 'dylmfcc', 'test', scalar_dylmfcc)
    normalization(dataset, 'dymspec', 'test', scalar_dymspec)
    normalization(dataset, 'lmfcc', 'test', scalar_lmfcc)
    normalization(dataset, 'mspec', 'test', scalar_mspec)
    
    
normalization_all_datasets()