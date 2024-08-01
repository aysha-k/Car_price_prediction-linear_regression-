#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV


# In[51]:


data=pd.read_csv(r"C:\Users\pc\Desktop\CarPrice_Assignment.csv")
data


# In[52]:


data.head()


# In[55]:


data.isnull().sum()


# In[56]:


data.info()


# In[59]:


data.describe().T


# In[61]:


data.shape


# In[75]:


data.dtypes


# In[116]:


data.drop(["car_ID"],inplace=True)


# In[62]:


data['doornumber'].value_counts()


# In[67]:


sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(data['price'], bins=30, kde=True, color='blue')
plt.title('Distribution of Car Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# In[68]:


data['doornumber'].unique


# In[71]:


mappings = {
    'doornumber': {'two': 2, 'four': 4},
    'fueltype': {'gas': 2, 'diesel': 1},
    'aspiration': {'std': 1, 'turbo': 2},
    'enginelocation': {'front': 1, 'rear': 2},
    'cylindernumber': {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'eight': 8, 'twelve': 12},
    'drivewheel': {'rwd': 1, 'fwd': 2, '4wd': 3}
}

data.replace(mappings, inplace=True)


# In[70]:


data


# In[72]:


data.head()


# In[117]:


data.dtypes


# In[118]:


lr=LabelEncoder()
data['carbody']=lr.fit_transform(data['carbody'])
data['fuelsystem']=lr.fit_transform(data['fuelsystem'])
data['enginetype']=lr.fit_transform(data['enginetype'])
data


# In[119]:


data.dtypes


# In[120]:


X = data.iloc[:,:-1]
y = data.iloc[:,:-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=100)


# In[121]:


X_train.shape


# In[122]:


X_test.shape


# In[123]:


y_train.shape


# In[124]:


y_test.shape


# In[125]:


lr= LinearRegression()


# In[127]:


lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)


# In[128]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[129]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,y_pred)


# In[139]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
sns.lineplot(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values with Line of Best Fit")
plt.show()

