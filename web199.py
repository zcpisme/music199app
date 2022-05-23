# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:44:42 2022

@author: 12427
"""

import numpy as np
import pandas as pd
import streamlit as st

# Start writing code here...
import pandas as pd
import numpy as np
import altair as alt

from sklearn.model_selection import train_test_split

from numpy import mean
from numpy import absolute
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error

import joblib
import pickle

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

linear = joblib.load('linear.pkl')
knn = joblib.load('knn.pkl')
rf = joblib.load('rf.pkl')

genre = ['rnb', 'rap', 'electronic', 'rock', 'new age', 'classical', 'reggae',
       'blues', 'country', 'world', 'folk', 'easy listening', 'jazz', 'vocal',
       'children\'s', 'punk', 'alternative', 'spoken word', 'pop',
       'heavy metal']

prediction_model = ('Linear','KNN','Random Forest')

with open('democol', 'rb') as fp:
    col = pickle.load(fp)
    

mycountry = col[1:-2]
mycountry2 = []
for i in mycountry:
    mycountry2.append(i[-2:])
mycountry_dict = dict(zip(mycountry2,mycountry))    

	
def mypredict(age = 22,country = 'US',gender = 'm', model = 'KNN'):

    genre = ['rnb', 'rap', 'electronic', 'rock', 'new age', 'classical', 'reggae',
       'blues', 'country', 'world', 'folk', 'easy listening', 'jazz', 'vocal',
       'children\'s', 'punk', 'alternative', 'spoken word', 'pop',
       'heavy metal']

    mydf = pd.DataFrame(columns = col)
    mydf.loc[0] =0


    mydf.loc[0,'age'] = age
    mydf.loc[0,'coun_'+country] = 1
    mydf.loc[0,'gender_'+gender] = 1


    prob_linear = linear.predict(mydf).reshape(-1)
    prob_knn = knn.predict(mydf).reshape(-1)
    prob_rf = rf.predict(mydf).reshape(-1)

    def mySort(sub_li):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of 
    # sublist lambda has been used
        sub_li.sort(key = lambda x: x[1],reverse = True)
        return sub_li

    if model == 'Linear':
        st.write('by Linear Model, the recommendation probability are:')
        ascending = mySort(list(zip(genre,prob_linear)))
        for a,b in ascending:
            st.write(a,':',"{:.3f}".format(b*100),'%')

    if model == 'KNN':
        st.write('\n by KNN Model, the recommendation probability are:')
        ascending = mySort(list(zip(genre,prob_knn)))
        for a,b in ascending:
            st.write(a,':',"{:.3f}".format(b*100),'%')

    if model == 'Random Forest':
        st.write('\n by RF Model, the recommendation probability are:')
        ascending = mySort(list(zip(genre,prob_rf)))
        for a,b in ascending:
            st.write(a,':',"{:.3f}".format(b*100),'%')

    return 'recommendation ends'




age = st.slider('Select Your Age?', min_value=4,max_value=120)
gender = st.selectbox('Your Gender is?', ('Female','Male'))
if gender == 'Female':
    gender = 'f'
else:
    gender = 'm'
country = st.selectbox('Your Country is:', mycountry2)


choose_model = st.selectbox(('Which model is used?'), prediction_model)

st.write(mypredict(age = age,country=country, gender = gender, model=choose_model))

