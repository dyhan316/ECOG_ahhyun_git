import numpy as np 
from scipy import signal, stats
import pandas as pd

#import models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVC, SVR, LinearSVC, NuSVC
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import os 

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--subject_agg', type = str, default = 'sub_agg', choices = ['sub_agg', 'sub_wise'], help = 'sub_wise or sub_agg')
parser.add_argument('--subject_num', type = int, default = None, help = 'subject number, either 0,1,2')

args = parser.parse_args()


assert (0<= args.seed) & (args.seed <= 49), "seed (i.e. mask num) must be between 0 and 49"


################################
#############SETUPS#############
################################
#model to test setup
classifiers_to_test = {
    "LinearRegression": LinearRegression(), #-inf~inf, so need to partition based on 0.5 #trash
    "Ridge": Ridge(), #-inf~inf, so need to partition based on 0.5 #trash
    "SVC": SVC(C=1e3), #0 or 1 #this works well when C is 1e3, 
    "LinearSVC": LinearSVC(), #0 or 1  #trash
    # "NuSVC": NuSVC(), #0 or 1  #trash
    "RandomForestClassifier": RandomForestClassifier(), #0 or 1 #good 
    "RandomForestClassifier_1000trees": RandomForestClassifier(n_estimators=1000), #very good 
    "DecisionTreeClassifier": DecisionTreeClassifier(), #0 or 1 #trash
    "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=84), 
    "GradientBoostingClassifier": GradientBoostingClassifier(),
    "GaussianMixture": GaussianMixture(n_components=2), 
    "BayesianGaussianMixture": BayesianGaussianMixture(n_components=2), 
    "ElasticNet": ElasticNet(), 
    "MLP32" :  MLPClassifier(hidden_layer_sizes=(32), validation_fraction=0.3,
                        learning_rate='adaptive', alpha=0.001, batch_size=32,
                        early_stopping=True, max_iter=1000, random_state=42),    
    "MLP64" :  MLPClassifier(hidden_layer_sizes=(64), validation_fraction=0.3,
                        learning_rate='adaptive', alpha=0.001, batch_size=32,
                        early_stopping=True, max_iter=1000, random_state=42),    
    "MLP128" :  MLPClassifier(hidden_layer_sizes=(128), validation_fraction=0.3,
                        learning_rate='adaptive', alpha=0.001, batch_size=32,
                        early_stopping=True, max_iter=1000, random_state=42),    
    "MLP256" :  MLPClassifier(hidden_layer_sizes=(256), validation_fraction=0.3,
                        learning_rate='adaptive', alpha=0.001, batch_size=32,
                        early_stopping=True, max_iter=1000, random_state=42),    
    "MLP512" :  MLPClassifier(hidden_layer_sizes=(512), validation_fraction=0.3,
                        learning_rate='adaptive', alpha=0.001, batch_size=32,
                        early_stopping=True, max_iter=1000, random_state=42),    
    "MLP128-128" : MLPClassifier(hidden_layer_sizes=(128,128), validation_fraction=0.3,
                        learning_rate='adaptive', alpha=0.001, batch_size=32,
                        early_stopping=True, max_iter=1000, random_state=42),    
}

if args.subject_agg == 'sub_agg':
    #added only for sub_agg cuz they require more samples
    classifiers_to_test.update({
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=100), #0 or 1 #very good 
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=200), #0 or 1 #very good 
    })

#datasets to test setup
base_config = {'data': {'dataset_name': '3ppl', 'base_path': '/scratch/connectome/dyhan316/ECOG_PILOT/data_rearranged', 
                        'mask_num': args.seed, 'subject_agg': args.subject_agg}}
if base_config['data']['subject_agg'] != 'sub_agg':
    assert args.subject_num is not None, "subject number must be given"


additional_config_components_to_test_dict = {
    #timeseries
    "raw" : {'norm': 'raw', 'norm_type': 'None', 'stft_type': 'None'},
    "fixation_denormed" : {'norm': 'norm', 'norm_type': 'fixation_denormed', 'stft_type': 'None'},
    "fixation_normalized" : {'norm': 'norm', 'norm_type': 'fixation_normalized', 'stft_type': 'None'},
    "whole_trial_denormed" : {'norm': 'norm', 'norm_type': 'whole_trial_denormed', 'stft_type': 'None'},
    "whole_trial_normalized" : {'norm': 'norm', 'norm_type': 'whole_trial_normalized', 'stft_type': 'None'},
    #stft
    'base' : {'norm': 'raw', 'norm_type': 'None', 'stft_type': 'base'},
    'halfnfft' : {'norm': 'raw', 'norm_type': 'None', 'stft_type': 'halfnfft'},
    'halfoverlap_halfnfft' : {'norm': 'raw', 'norm_type': 'None', 'stft_type': 'halfoverlap_halfnfft'},
    'halfoverlap' : {'norm': 'raw', 'norm_type': 'None', 'stft_type': 'halfoverlap'},
    'nooverlap_halfnfft' : {'norm': 'raw', 'norm_type': 'None', 'stft_type': 'nooverlap_halfnfft'},
    'nooverlap' : {'norm': 'raw', 'norm_type': 'None', 'stft_type': 'nooverlap_halfnfft'},
}

config_dict = {}
for data_name, additional_config_components in additional_config_components_to_test_dict.items():
    config = base_config.copy()
    config['data'].update(additional_config_components)
    config_dict[data_name] = config

################################
#########Preproc Data###########
################################
from data_utils import get_train_test_data
train_data_dict = {}
train_labels_dict = {}
test_data_dict = {}
test_labels_dict = {}
for config_name, config in config_dict.items():
    train_data, test_data, train_label, test_label = get_train_test_data(config['data'], subject = args.subject_num)
    train_data_dict[config_name] = train_data
    train_labels_dict[config_name] = train_label
    test_data_dict[config_name] = test_data
    test_labels_dict[config_name] = test_label
    print(f"{config_name} completed")
    
# #assume these are already given
# # train_data_dict, test_data_dict, train_labels_dict, test_labels_dict = 

# ##testing thing 


# train_thing = np.load('/scratch/connectome/dyhan316/ECOG_PILOT/data_rearranged/subject-aggregated/raw_timeseries_data.npy')
# test_thing = np.load('/scratch/connectome/dyhan316/ECOG_PILOT/data_rearranged/subject-aggregated/raw_timeseries_label.npy')
# X_train, X_test, y_train, y_test = train_test_split(train_thing, test_thing, test_size=0.2, random_state=args.seed)
# train_data_dict = {'raw_timeseries': X_train}
# train_labels_dict = {'raw_timeseries': y_train}
# test_data_dict = {'raw_timeseries': X_test}
# test_labels_dict = {'raw_timeseries': y_test}



#! TODO : do stuff like flattening and stuff! (stft and such!)
#! ultimately get train_data_dict and such! 

################################
#####RUNNING WITHOUT PCA########
################################
performance_records = []


for model_name, model in classifiers_to_test.items():
    for data_name in train_data_dict.keys():
        print(f"Running {model_name} on {data_name}")
        train_data = train_data_dict[data_name]
        train_labels = train_labels_dict[data_name]
        test_data = test_data_dict[data_name]
        test_labels = test_labels_dict[data_name]
        
        
        
        ##training and running inference 
        model.fit(train_data, train_labels)
        ##!add the thresholding if needed  => check! 
        preds = model.predict(test_data)
        
        if model_name in ['LinearRegression', 'Ridge', 'ElasticNet']:
            preds = (preds>0.5).astype(int)
        ##saving the performances
        performance_records.append({
            'model_name': model_name,
            'data_name': data_name,
            'accuracy': accuracy_score(test_labels, preds),
            'roc_auc': roc_auc_score(test_labels, preds),
            'f1_score': f1_score(test_labels, preds),
            'balanced_accuracy': balanced_accuracy_score(test_labels, preds)
        })

################################
######RUNNING WITH PCA##########
################################
for n_components in [5, 10, 20, 50] : #! 200 will not work if subject-wise! (cuz not enough samples) 
    for model_name, model in classifiers_to_test.items():
        for data_name in train_data_dict.keys():
            train_data = train_data_dict[data_name]
            train_labels = train_labels_dict[data_name]
            test_data = test_data_dict[data_name]
            test_labels = test_labels_dict[data_name]
            
            ##PCA-ize the data
            pca = PCA(n_components=n_components)
            train_data = pca.fit_transform(train_data)
            test_data = pca.transform(test_data)
            
            ##training and running inference 
            model.fit(train_data, train_labels)
            ##!add the thresholding if needed  => check! 
            preds = model.predict(test_data)
            
            if model_name in ['LinearRegression', 'Ridge', 'ElasticNet']:
                preds = (preds>0.5).astype(int)
            
            ##saving the performances
            performance_records.append({
                'model_name': model_name,
                'data_name': f'PCA_{n_components}-'+data_name,
                'accuracy': accuracy_score(test_labels, preds),
                'roc_auc': roc_auc_score(test_labels, preds),
                'f1_score': f1_score(test_labels, preds),
                'balanced_accuracy': balanced_accuracy_score(test_labels, preds)
            })
results = pd.DataFrame(performance_records)
print(results)

#save the results
save_path_name = f'results/ML/{args.subject_agg}' 
save_path_name += '/subject-'+str(args.subject_num) if args.subject_agg != 'sub_agg' else ''
os.makedirs(save_path_name, exist_ok=True)
results.to_csv(f"{save_path_name}/results_seed-{args.seed}.csv", index=False)


##get 
#3. Bandpower (alpha, beta, gamma, theta, delta)
#4. subject-wise thing too! 
#cv?  => 일단 높은 성능이 필요하면! 

#during all subject vs subject-wise model comparison, keep the samples same!! 







