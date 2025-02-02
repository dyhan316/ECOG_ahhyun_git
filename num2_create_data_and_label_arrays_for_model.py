import os 
import numpy as np
from data_transforms import *

##! will sassume data of shape (subject, subtype, trial, channel, time), (subject, subtype, label)
def load_data_and_labels(data_arr_path, label_arr_path):
    return np.load(data_arr_path), np.load(label_arr_path)


def split_data():
    ...

#assumes sub is on 0 dim
def get_sub_only_keep_dim(data, label, sub_idx):
    sub_only_data = data[sub_idx]
    sub_only_label = label[sub_idx]
    return np.expand_dims(sub_only_data, axis=0), np.expand_dims(sub_only_label, axis=0)
    

def flatten_dimensions(data , label, dim_str_list):
    #dim_str options : ['subject', 'channel', 'subtype']
    if set(dim_str_list) == {'subject', 'channel', 'subtype'}:
        return data.reshape(-1, data.shape[-1]), label.flatten()  
    elif set(dim_str_list) == {'channel', 'subtype'}: 
        return data.reshape(data.shape[0], -1, data.shape[-1]), label.reshape(label.shape[0], -1)
    else : 
        raise NotImplementedError(f"flatten_dimensions not implemented for {dim_str_list}")
    
    
###STFT STUFF########
base_stft_params = {'fs' : 2000, 'clip_fs' : 150, "window" : 'hamming',"nperseg" : 2000/10,"noverlap" : 2000/10*0.9,"nfft" : 2000}
nooverlap_stft_params = {'fs' : 2000, 'clip_fs' : 150, "window" : 'hamming',"nperseg" : 2000/10,"noverlap" : 0,"nfft" : 2000} #removed noverlap
halfoverlap_stft_params = {'fs' : 2000, 'clip_fs' : 150, "window" : 'hamming',"nperseg" : 2000/10,"noverlap" : 2000/10*0.5,"nfft" : 2000} #removed noverlap
halfnfft_stft_params = {'fs' : 2000, 'clip_fs' : 150, "window" : 'hamming',"nperseg" : 2000/10,"noverlap" : 2000/10*0.9,"nfft" : 2000/2} #removed noverlap
halfoverlap_halfnfft_stft_params = {'fs' : 2000, 'clip_fs' : 150, "window" : 'hamming',"nperseg" : 2000/10,"noverlap" : 2000/10*0.5,"nfft" : 2000/2} #removed noverlap
nooverlap_halfnfft_stft_params = {'fs' : 2000, 'clip_fs' : 150, "window" : 'hamming',"nperseg" : 2000/10,"noverlap" : 0,"nfft" : 2000/2} #removed noverlap

def save_stft_f_t_and_Zxx(f, t, Zxx, save_path, file_name):
    np.save(os.path.join(save_path, 'stft', f'{file_name}-f.npy'), f)
    np.save(os.path.join(save_path, 'stft', f'{file_name}-t.npy'), t)
    np.save(os.path.join(save_path, 'stft', f'{file_name}-Zxx.npy'), Zxx)


######################

###! =====CREATE data arrays to test ===== !###
#get the arrayized final arrays
data = np.load('./data_rearranged/final_data_array.npy')
label = np.load('./data_rearranged/final_label_array.npy')


##########################################################
##########1. Subject-wise data and label arrays ##########
##########################################################
data_flattened, label_flattened = flatten_dimensions(data, label, ['channel', 'subtype'])
print("subject-wise data and labels", data_flattened.shape, label_flattened.shape) #shape (18, 200000) (18, 20) #(subject*channel*subtype, time), (subject*channel*subtype, label)

file_name = './data_rearranged/subject-wise/'


os.makedirs(f'{file_name}/normalized_timeseries', exist_ok = True)
os.makedirs(f'{file_name}/stft', exist_ok = True)
np.save(f'{file_name}/raw_timeseries_label.npy', label_flattened)

#save raw time series
np.save(f'{file_name}/raw_timeseries_data.npy', data_flattened)

#get normalized (four types)
np.save(f'{file_name}/normalized_timeseries/whole_trial_normalized_timeseries_data.npy', normalize_memory_task(data_flattened, option = "full_trial"))
np.save(f'{file_name}/normalized_timeseries/whole_trial_denormed_timeseries_data.npy', normalize_memory_task(data_flattened, demean_only = True, option = "full_trial"))
np.save(f'{file_name}/normalized_timeseries/fixation_normalized_timeseries_data.npy', normalize_memory_task(data_flattened, option = "fixation"))
np.save(f'{file_name}/normalized_timeseries/fixation_denormed_timeseries_data.npy', normalize_memory_task(data_flattened, demean_only = True, option = "fixation"))

#get STFT version
f, t, Zxx = parallel_get_stft(data_flattened, **base_stft_params)
save_stft_f_t_and_Zxx(f, t, Zxx, file_name, 'base_stft_timeseries_data')

f, t, Zxx = parallel_get_stft(data_flattened, **nooverlap_stft_params)
save_stft_f_t_and_Zxx(f, t, Zxx, file_name, 'nooverlap_stft_timeseries_data')

f, t, Zxx = parallel_get_stft(data_flattened, **halfoverlap_stft_params)
save_stft_f_t_and_Zxx(f, t, Zxx, file_name, 'halfoverlap_stft_timeseries_data')

f, t, Zxx = parallel_get_stft(data_flattened, **halfnfft_stft_params)
save_stft_f_t_and_Zxx(f, t, Zxx, file_name, 'halfnfft_stft_timeseries_data')

f, t, Zxx = parallel_get_stft(data_flattened, **halfoverlap_halfnfft_stft_params)
save_stft_f_t_and_Zxx(f, t, Zxx, file_name, 'halfoverlap_halfnfft_stft_timeseries_data')

f, t, Zxx = parallel_get_stft(data_flattened, **nooverlap_halfnfft_stft_params)
save_stft_f_t_and_Zxx(f, t, Zxx, file_name, 'nooverlap_halfnfft_stft_timeseries_data')



#! also save the max of the freq (250Hz 까지만 본다던지... 1000Hz is too large!)



################################################################
##########2. Subject Aggregated data and label arrays ##########
################################################################
data_flattened, label_flattened = flatten_dimensions(data, label, ['subject', 'channel', 'subtype'])   
print("subject-aggregated data and labels", data_flattened.shape, label_flattened.shape) #shape (18, 200000) (18, 20) #(subject*channel*subtype, time), (subject*channel*subtype, label)

file_name = './data_rearranged/subject-aggregated/'

os.makedirs(f'{file_name}/normalized_timeseries', exist_ok = True)
os.makedirs(f'{file_name}/stft', exist_ok = True)
np.save(f'{file_name}/raw_timeseries_label.npy', label_flattened)

#save raw time series
np.save(f'{file_name}/raw_timeseries_data.npy', data_flattened)

#get normalized (four types)
np.save(f'{file_name}/normalized_timeseries/whole_trial_normalized_timeseries_data.npy', normalize_memory_task(data_flattened, option = "full_trial"))
np.save(f'{file_name}/normalized_timeseries/whole_trial_denormed_timeseries_data.npy', normalize_memory_task(data_flattened, demean_only = True, option = "full_trial"))
np.save(f'{file_name}/normalized_timeseries/fixation_normalized_timeseries_data.npy', normalize_memory_task(data_flattened, option = "fixation"))
np.save(f'{file_name}/normalized_timeseries/fixation_denormed_timeseries_data.npy', normalize_memory_task(data_flattened, demean_only = True, option = "fixation"))

#get STFT version
f, t, Zxx = parallel_get_stft(data_flattened, **base_stft_params)
save_stft_f_t_and_Zxx(f, t, Zxx, file_name, 'base_stft_timeseries_data')

f, t, Zxx = parallel_get_stft(data_flattened, **nooverlap_stft_params)
save_stft_f_t_and_Zxx(f, t, Zxx, file_name, 'nooverlap_stft_timeseries_data')

f, t, Zxx = parallel_get_stft(data_flattened, **halfoverlap_stft_params)
save_stft_f_t_and_Zxx(f, t, Zxx, file_name, 'halfoverlap_stft_timeseries_data')

f, t, Zxx = parallel_get_stft(data_flattened, **halfnfft_stft_params)
save_stft_f_t_and_Zxx(f, t, Zxx, file_name, 'halfnfft_stft_timeseries_data')

f, t, Zxx = parallel_get_stft(data_flattened, **halfoverlap_halfnfft_stft_params)
save_stft_f_t_and_Zxx(f, t, Zxx, file_name, 'halfoverlap_halfnfft_stft_timeseries_data')

f, t, Zxx = parallel_get_stft(data_flattened, **nooverlap_halfnfft_stft_params)
save_stft_f_t_and_Zxx(f, t, Zxx, file_name, 'nooverlap_halfnfft_stft_timeseries_data')


######################################################################################
##########3. SAVE INDICES USED FOR TRAINING AND TESTING FOR REPRODUCIBILITY ##########
######################################################################################
#two should be saved : one for subject-wise, one for subject-aggregated
#the indices should be different shape for each of the two (cuz the data is different shape)
#for subject-wise regardless of whether it's time-wise or stft-wise the shape is like (3, 120, ...)
#for subject-aggregated regardless of whether it's time-wise or stft-wise the shape is like (360, ...)
#our goal here is to save the np array of masks that can be used on the original data to retrieve the data used for training and testing
#the mask should be of shape (360, ) for subject-aggregated, and (3, 120, ...) for subject-wise

TEST_RATE = 0.3
NUMBER_OF_MASKS_TO_SAVE = 50

#subject-wise
mask_list = []
for seed in range(NUMBER_OF_MASKS_TO_SAVE):
    np.random.seed(seed)
    mask = np.zeros((3, 120), dtype=bool)
    for i in range(3):
        indices = np.random.choice(120, int(120 * TEST_RATE), replace=False)
        mask[i, indices] = True
    file_name = './data_rearranged/subject-wise/'
    os.makedirs(f'{file_name}/masks', exist_ok = True)
    np.save(f'{file_name}/masks/mask_{seed}.npy', mask)
    mask_list.append(mask)

#subject-aggregated
for seed in range(NUMBER_OF_MASKS_TO_SAVE):
    file_name = './data_rearranged/subject-aggregated/'
    os.makedirs(f'{file_name}/masks', exist_ok = True)
    np.save(f'{file_name}/masks/mask_{seed}.npy', mask_list[seed].flatten())
    
    # np.random.seed(seed)
    # mask = np.zeros(360, dtype=bool)
    # indices = np.random.choice(360, int(360 * TEST_RATE), replace=False)
    # mask[indices] = True
    # file_name = './data_rearranged/subject-aggregated/'
    # os.makedirs(f'{file_name}/masks', exist_ok = True)
    # np.save(f'{file_name}/masks/mask_{seed}.npy', mask)



# data_flattened, label_flattened = flatten_dimensions(data, label, ['subject', 'channel', 'subtype'])
data_flattened, label_flattened = flatten_dimensions(data, label, ['subject', 'channel', 'subtype'])
data_flattened = Zxx.reshape(data_flattened.shape[0], -1) #flatten the stft #overwritten.. shouldn't be like this!
print(data_flattened.shape, label_flattened.shape) 

print(np.unique(label_flattened))

#! remove later!
# data, label = get_sub_only_keep_dim(data, label, 2)

#0.5921296296296297 0.5897750257197281
#0.6027777777777779 0.5057733266905093
#0.5666666666666667 0.5367227888067672
#0.6347222222222223 0.4848814540436058

#save
# np.save('./data_rearranged/all_mashed/final_data_all_flattened.npy', data_flattened)
# np.save('./data_rearranged/all_mashed/final_label_all_flattened.npy', label_flattened)




#run linear regression on the things 
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVC, SVR, LinearSVC, NuSVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
# Split the data into training and testing sets
acc_list = []
roc_auc_list = []
f1_list = []
for seed in range(30) : 
    print(seed)
    X_train, X_test, y_train, y_test = train_test_split(data_flattened, label_flattened, test_size=0.2, random_state=seed)

    # Initialize and train the linear regression model
    model = GradientBoostingClassifier(validation_fraction= 0.3 ) #default :  0.5944444444444444
    #RandomForestClassifier()
    #LinearSVC()
    #Ridge(alpha=0.1)
    ##LinearCl()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    
    print(y_pred.mean(), y_test.mean(), np.unique(y_pred), np.unique(y_test))
    print(y_train.mean(), model.predict(X_train).mean())

    # Calculate accuracy and ROC AUC score
    # accuracy = accuracy_score(y_test, np.round(y_pred)) #this should be chagned if differnet model
    accuracy = accuracy_score(y_test, (y_pred>0.5).astype(int)) #this should be chagned if differnet model
    roc_auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, (y_pred>0.5).astype(int))

    # print(f'Accuracy: {accuracy}')
    # print(f'ROC AUC Score: {roc_auc}')
    acc_list.append(accuracy)
    roc_auc_list.append(roc_auc)
    f1_list.append(f1)

print(np.mean(acc_list), np.std(acc_list)) #0.5 0.5
print(np.mean(roc_auc_list), np.std(roc_auc_list)) #0.0 0.0
print(np.mean(f1_list), np.std(f1_list)) 




""" 
RandomForestClassifier
0.6129629629629629 0.03807571332398534
0.6103168036028371 0.03774117249577359
0.6596337409676619 0.04845509470313719

RandomForestClassifier with 1000 trees
0.6199074074074075 0.03820215971458565
0.6160989264035205 0.038150609779393524
0.6711224366309287 0.04481030765988914

RandomForestClassifier with 3000 trees
0.6185185185185186 0.03807571332398534
0.6149934074855016 0.03846370324900992
0.6696593651542403 0.04377298321643469

RandomForestClassifier(n_estimators=1000, criterion='entropy')
0.6189814814814816 0.037039930442536305
0.6152545570065195 0.03766666948754423
0.6701388361322358 0.04347555316628227

RandomForestClassifier(n_estimators=1000, criterion='log_loss')
0.6189814814814816 0.037039930442536305
0.6152545570065195 0.03766666948754423
0.6701388361322358 0.04347555316628227

Ridge 0.1
0.5111111111111111 0.053117812997272085
0.5260930420474598 0.0615140010625201
0.6283304785973463 0.05640379366053577

Ridge 0.5 
0.5111111111111111 0.053117812997272085
0.5259892744563476 0.06144102771215337
0.6283304785973463 0.05640379366053577

LinearRegression
0.5111111111111111 0.053117812997272085
0.5259879909246068 0.0615600052858613
0.6283304785973463 0.05640379366053577

LinearSVC
0.5027777777777778 0.053599835697801135
0.5043980060777549 0.04794721398270588
0.5028489150937175 0.11320944518756948

GradientBoostingClassifier
0.6018518518518517 0.050248013152648645
0.5989086863509999 0.04651479004214873
0.6473487313140325 0.05640656630530396

NuSVC
0.5249999999999999 0.09738082301643392
0.524987683411287 0.09332064992616032
0.5653695993515934 0.13384997894966405

DecisionTreeClassifier
0.5217592592592593 0.06008003635298154
0.5223829954610244 0.061832551818816806
0.5463588547011111 0.06097839862448603

KNeighborsClassifier
0.5333333333333334 0.04400827643820794
0.5283802298854586 0.041059898261785195
0.5908492783502859 0.049448409431738324

GaussianMixture
0.48009259259259257 0.05143058035192755
0.5 0.0
0.0 0.0

BayesianGaussianMixture
0.48009259259259257 0.05143058035192755
0.5 0.0
0.0 0.0
"""
