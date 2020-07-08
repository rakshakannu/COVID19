import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
df = pd.read_csv('dataset.csv')

def data_split(data,ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data)*ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__ == 'main':
    df = pd.read_csv('dataset.csv')
    train, test = data_split(df, 0.2)
    X_train = train[['fever','bodypain','age','cough','diffbreathing']].to_numpy()
    X_test = test[['fever','bodypain','age','cough','diffbreathing']].to_numpy()

    Y_train = train[['Covid19']].to_numpy().reshape(1606 ,)
    Y_test = test[['Covid19']].to_numpy().reshape(401 ,1)

    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression()
    clf.fit(X_train, Y_train)

    with open ('model_pickle','wb') as f:
        pickle.dump(clf,f)

    #code for inference
    inputfeatures = [100,1,22,1,1]
    infprob = clf.predict_proba([inputfeatures])[0][1]

