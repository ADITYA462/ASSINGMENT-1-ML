#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


url = "https://raw.githubusercontent.com/edyoda/data-science-complete-tutorial/master/Data/house_rental_data.csv.txt"
df = pd.read_csv(url)


# # Use pandas to get some insights into the data

# In[9]:


df.head()


# In[10]:


df.reset_index(drop = True)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


# In[11]:


df.tail()


# In[12]:


df.dtypes


# In[13]:


df.describe()


# In[14]:


x = df.shape
print("\n shape of: \n",x)

y = df.size
print("\n size of : \n",y)


# In[15]:


df.rename(columns = {"Living.Room" : "Living_room"})


# In[16]:


df[df["Sqft"] >= 300.000]


# In[17]:


df.info()


# In[18]:


df.isnull().count()


# In[19]:


df["Sqft"].value_counts()


# In[20]:


df["Bathroom"].value_counts()


# # Show some interesting visualization of the data

# In[25]:


sns.displot(df['Sqft'],kde = True,color = "r" )


# In[22]:


sns.heatmap(df.corr(), annot = True,cmap=plt.cm.get_cmap('viridis_r'))


# In[26]:


sns.lineplot(x = df["Sqft"], y = df["Bathroom"])


# In[27]:


sns.pairplot(data = df,hue = "Price",palette = 'coolwarm')


# In[28]:


sns.scatterplot(data=df,x = df['Sqft'],y = df["Price"],hue = "TotalFloor")


# # Manage data for training & testing

# In[30]:


from scipy import stats
x = np.abs(stats.zscore(df))
x


# In[31]:


X = df.drop(labels = ['Price'] , axis = 1)
y = df['Price']


# In[32]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


# In[33]:


scaler = MinMaxScaler(feature_range = (0,1))
X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)

X_test_scaled = scaler.fit_transform(X_test)
X_test = pd.DataFrame(X_test_scaled)


# In[34]:


from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
from sklearn.neighbors import KNeighborsClassifier


# In[35]:


rmse = []
for k in range(1,21):
    knn = neighbors.KNeighborsRegressor(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error = sqrt(mean_squared_error(y_test,y_pred))
    rmse.append(error)
    
    print('RMSE for k =',k,'is',error)


# In[36]:


curve = pd.DataFrame(rmse)
curve.plot()


# In[ ]:




