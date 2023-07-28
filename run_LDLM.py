import pickle
import os
import numpy as np
from ldlm import LDLM
from sklearn.model_selection import train_test_split


def save_dict(dataset, scores, name):
    with open(dataset + "//" + name, 'wb') as f:
        pickle.dump(scores, f)

def load_dict(dataset, name):
    file_name = dataset + "//" + name
    if not os.path.exists(file_name):
        file_name += ".pkl"
        
    with open(file_name, 'rb') as f:
        return pickle.load(f)
    
    
def expected_zero_one_loss(Y_pre, Y):
    Y_l = np.argmax(Y_pre, 1)
    return 1 - Y[np.arange(Y.shape[0]), Y_l].mean(0)


def zero_one_loss(Y_pre, Y):
    Y_l_pre = np.argmax(Y_pre, 1)
    Y_l = np.argmax(Y, 1)
    
    return 1 - (Y_l_pre == Y_l).mean()       

    

def run_LDLM(dataset, i, train_x, train_y, test_x, test_y):
    l1 = 0.001   
    l2 = 0.1 #need tuning l2 from [0.001, 0.01, 0.1, 1]
    l3 = 0.1 #need tuning l3 from [0.001, 0.01, 0.1, 1]
    rho = 0.01   
    
    model = LDLM(train_x, train_y, l1, l2, l3, rho)
    model.fit()
    y_pre = model.predict(test_x)
    return (zero_one_loss(y_pre, test_y), expected_zero_one_loss(y_pre, test_y))

    

def run_KF(dataset):
    print(dataset)
    X, Y = np.load(dataset + "//feature.npy"), np.load(dataset + "//label.npy")
    for i in range(10):
        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3)      
        losses = run_LDLM(dataset, i, train_x, train_y, test_x, test_y)
        print(losses)


if __name__ == "__main__":
    
    datasets = ["SJAFFE"]
    for dataset in datasets:
        run_KF(dataset)

