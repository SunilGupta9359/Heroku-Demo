import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

dataset=pd.read_csv("hiring.csv")
dataset['experience'].fillna(0, inplace=True)

dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

X=dataset.iloc[:,:3]

#convert word to integer value
def convert_to_int(word):
    word_dict={'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 
              'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0:0}
    return word_dict[word]

X['experience']=X['experience'].apply(lambda X:convert_to_int(X))

y=dataset.iloc[:,-1]


#splitting into training and testing dataset
#having small dataset so we will train our model using all dataset
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

#fitting model with training dataset
regressor.fit(X,y)

#Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

#loading model to compair the result 
model=pickle.load(open('model.pkl','rb'))
print(model.predict([[2,9,6]]))







