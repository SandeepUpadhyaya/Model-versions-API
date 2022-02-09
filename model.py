# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('hiring.csv')

dataset['experience'].fillna(0, inplace=True)

dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

X = dataset.iloc[:, :3]

#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
model1 = LinearRegression()

#Fitting model with trainig data
model1.fit(X, y)

# Saving model to disk
pickle.dump(model1, open('model1.pkl','wb'))

# Loading model to compare the results
model1 = pickle.load(open('model1.pkl','rb'))
print(model1.predict([[8,8,8]]))

# second version of the model
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
model2 = LinearRegression()

#Fitting model with trainig data
model2.fit(X, y)

# Saving model to disk
pickle.dump(model2, open('model2.pkl','wb'))

# Loading model to compare the results
mode2 = pickle.load(open('model2.pkl','rb'))
print(model2.predict([[2, 9, 8]]))
