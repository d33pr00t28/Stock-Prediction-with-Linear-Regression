#!/usr/bin/env python
# coding: utf-8

# # Trying to predict price of AMAZON using Linear Regression

# In[143]:


import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime as dt
import pandas_datareader.data as web
import math
from sklearn.model_selection import train_test_split


# In[144]:


from pandas import Series, DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# # Getting AMAZON data from YAHOO

# In[146]:


start = dt.datetime(2016, 1, 1)
end = dt.datetime(2019, 7, 1)


# In[147]:


df = web.DataReader("AMZN", 'yahoo', start, end)


# In[148]:


df.head()


# In[149]:


df.tail()


# # Plotting Closing Price of Amazon

# In[150]:


df['Adj Close'].plot(label='AMZN',figsize=(16,8),title='AMAZON')
plt.legend()


# # Calculating Rolling Mean & Plotting

# In[151]:


close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()


# In[152]:


mavg.tail()


# In[153]:


close_px.plot(label = 'AMZN',figsize=(16,8),title='AMAZON')
mavg.plot(label = 'MA')
plt.legend()


# # Calculating & Plotting the returns of Amazon

# In[154]:


rets = close_px / close_px.shift(1) - 1
rets.plot(label='return')


# # Predicting Stock Price using LR,QDA &  KNN

# # Define features: High Low Percentage and Percentage Change.

# In[155]:


df_reg = df.loc[:, ['Adj Close', 'Volume']]
df_reg['HL_PCT'] = (df['High'] - df['Low']) / df['Adj Close'] * 100.0
df_reg['PCT_change'] = (df['Adj Close'] - df['Open']) / df['Open'] * 100.0


# In[156]:


print(df_reg)


# #   Filling Missing Values

# In[157]:


df_reg.replace([np.inf, -np.inf], np.nan)

df_reg.dropna(inplace=True)


# In[158]:


df_reg.fillna(df_reg.mean(), inplace=True)


# #Retriving 1% data for learning 

# In[159]:


forecast_out = int(math.ceil(0.01 * len(df_reg)))


# # Separating labels, we want to predict for Adj Close.

# In[160]:


forecast_col = 'Adj Close'
df_reg['label'] = df_reg['Adj Close'].shift(forecast_out)
X = np.array(df_reg.drop(['label'], 1))


# # Scale X

# In[161]:


X = sk.preprocessing.scale(X)


# In[162]:


# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]


# In[163]:


# Separate label and identify it as y
y = np.array(df_reg['label'])
y = y[:-forecast_out]


# # Separation of training and testing of model by cross validation train test split

# In[164]:


X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.1, random_state=0)


# In[ ]:





# # Linear Regression

# In[165]:


X_train = np.nan_to_num(X_train)
y_train= np.nan_to_num(y_train)
X_test=np.nan_to_num(X_test)
y_test=np.nan_to_num(y_test)


# In[166]:


clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)


# # Quadratic Regression 2

# In[167]:


clf_poly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clf_poly2.fit(X_train, y_train)


# # Quadratic Regression 3

# In[168]:


clf_poly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clf_poly3.fit(X_train, y_train)


# # KNN

# In[169]:


clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)


# # Evaluate Models using Score Method

# In[170]:


confidence_reg = clfreg.score(X_test, y_test)
confidencepoly2 = clf_poly2.score(X_test,y_test)
confidencepoly3 = clf_poly3.score(X_test,y_test)
confidenceknn = clfknn.score(X_test, y_test)


# In[171]:


results = [confidence_reg,confidencepoly2,confidencepoly3,confidenceknn]


# In[172]:


print (results)


# In[173]:


forecast_set = clfreg.predict(X_lately)
df_reg['Forecast'] = np.nan


# In[174]:


last_date = df_reg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + dt.timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += dt.timedelta(days=1)
    df_reg.loc[next_date] = [np.nan for _ in range(len(df_reg.columns)-1)]+[i]
df_reg['Adj Close'].tail(500).plot(figsize = (16,8))
df_reg['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# In[102]:





# In[ ]:




