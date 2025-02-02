#train/test split must be done! 
#cv? 
import numpy as np
from pilotStudy_3p.package.utils import load_data, MyDataset

##CODE MADE TO USE AHHYUN'S THINGS###
def get_train_test_split(data, label, mask, subject = None) : 
    if subject is not None : 
        mask = mask[subject]
        data = data[subject]
        label = label[subject]
    train_indices = np.where(mask == 0)[0]
    test_indices = np.where(mask == 1)[0]
    
    train_data = data[train_indices]
    test_data = data[test_indices]
    
    train_label = label[train_indices]
    test_label = label[test_indices]
    
    return train_data, test_data, train_label, test_label


def get_train_test_data(data_config, subject = None) : 
    dataset = MyDataset(data_config)
    train_data, test_data, train_label, test_label = get_train_test_split(dataset.data, dataset.labels, dataset.mask, subject)
    #flatten the last two dims if STFT 
    #only for ML as we flatten stuff 
    if data_config['stft_type'] != "None":
        #may not work if the shape is not (n, m, l)
        train_data = train_data.reshape(train_data.shape[0], -1)
        test_data = test_data.reshape(test_data.shape[0], -1)    
    
    return train_data, test_data, train_label, test_label



"""backup before implementing subjects
def get_train_test_split(data, label, mask) : 
    train_indices = np.where(mask == 0)[0]
    test_indices = np.where(mask == 1)[0]
    
    train_data = data[train_indices]
    test_data = data[test_indices]
    
    train_label = label[train_indices]
    test_label = label[test_indices]
    
    return train_data, test_data, train_label, test_label


def get_train_test_data(data_config) : 
    dataset = MyDataset(data_config)
    train_data, test_data, train_label, test_label = get_train_test_split(dataset.data, dataset.labels, dataset.mask)
    import pdb ; pdb.set_trace()
    #flatten the last two dims if STFT 
    #only for ML as we flatten stuff 
    if data_config['stft_type'] != "None":
        #may not work if the shape is not (n, m, l)
        train_data = train_data.reshape(train_data.shape[0], -1)
        test_data = test_data.reshape(test_data.shape[0], -1)    
    
    return train_data, test_data, train_label, test_label



"""

################################

##! will sassume data of shape (subject, subtype, trial, channel, time), (subject, subtype, label)
def load_data_and_labels(data_arr_path, label_arr_path):
    return np.load(data_arr_path), np.load(label_arr_path)


#assumes sub is on 0 dim
def get_sub_only_keep_dim(data, label, sub_idx):
    sub_only_data = data[sub_idx]
    sub_only_label = label[sub_idx]
    return np.expand_dims(sub_only_data, axis=0), np.expand_dims(sub_only_label, axis=0)
    

def flatten_dimensions(data, label, dim_str_list):
    #dim_str options : ['subject', 'channel', 'subtype']
    if set(dim_str_list) == {'subject', 'channel', 'subtype'}:
        return data.reshape(-1, data.shape[-1]), label.flatten()  
        ##! MAKE SURE THE LABELS ARE IN THE SAME ORDER AS THE DATA (HOW? I DONT KNOW)
        
    else : 
        raise NotImplementedError("flatten_dimensions only implemented for ['subject', 'channel', 'subtype']")


##testing 
if __name__ == "__main__":
    ###CREATE data arrays to test 

    #test mash_dimensions
    data = np.load('./data_rearranged/final_data_array.npy')
    label = np.load('./data_rearranged/final_label_array.npy')



    data_flattened, label_flattened = flatten_dimensions(data, label, ['subject', 'channel', 'subtype'])
    print(data_flattened.shape, label_flattened.shape) #shape (18, 200000) (18, 20) #(subject*channel*subtype, time), (subject*channel*subtype, label)



    #! remove later!
    data, label = get_sub_only_keep_dim(data, label, 2)


    #0.5921296296296297 0.5897750257197281
    #0.6027777777777779 0.5057733266905093
    #0.5666666666666667 0.5367227888067672
    #0.6347222222222223 0.4848814540436058

    #save
    # np.save('./data_rearranged/final_data_all_flattened.npy', data_flattened)
    # np.save('./data_rearranged/final_label_all_flattened.npy', label_flattened)




    #run linear regression on the things 
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.svm import SVC, SVR, LinearSVC

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.ensemble import RandomForestClassifier
    # Split the data into training and testing sets
    acc_list = []
    roc_auc_list = []
    for seed in range(30) : 
        print(seed)
        X_train, X_test, y_train, y_test = train_test_split(data_flattened, label_flattened, test_size=0.2, random_state=seed)

        # Initialize and train the linear regression model
        model = RandomForestClassifier()
        #LinearSVC()
        #Ridge(alpha=0.1)
        ##LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate accuracy and ROC AUC score
        accuracy = accuracy_score(y_test, np.round(y_pred))
        roc_auc = roc_auc_score(y_test, y_pred)

        # print(f'Accuracy: {accuracy}')
        # print(f'ROC AUC Score: {roc_auc}')
        acc_list.append(accuracy)
        roc_auc_list.append(roc_auc)

    print(np.mean(acc_list), np.mean(roc_auc_list)) #0.5 0.5
    print(np.std(acc_list), np.std(roc_auc_list)) #0.0 0.0





