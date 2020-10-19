import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
features = pd.read_csv('A2.csv')
features = pd.get_dummies(features)
flag = np.array(features['Modal Price (Rs./Quintal)'])
features = features.drop('Modal Price (Rs./Quintal)',axis=1)
features_list = list(features.columns)
features=np.array(features)
X_train,X_test,Y_train,Y_test = train_test_split(features,flag,test_size=0.25,random_state=42)
#print('Training Features Shape:', X_train.shape)
#print('Training Labels Shape:', Y_train.shape)
#print('Testing Features Shape:', X_test.shape)
#print('Testing Labels Shape:', Y_test.shape)
rf = RandomForestRegressor(n_estimators=1000,random_state=42)
rf.fit(X_train,Y_train)
pred = rf.predict(X_test)
error = abs(pred - Y_test)
quil = 100*(np.sum(error)/np.sum(Y_test))
accuracy = 100 - np.mean(quil)
print(accuracy,"%")