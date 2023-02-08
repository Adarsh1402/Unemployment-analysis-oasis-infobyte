#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries. 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
     


# In[2]:


data = pd.read_csv("Unemployment in India.csv")


# In[25]:


data.head()


# In[26]:


#change coloum name for understanding
data.columns= ["States","Date","Frequency",
               "Estimated Unemployment Rate",
               "Estimated Employed",
               "Estimated Labour Participation Rate",
               "Region"]
     


# In[27]:


data.head()


# In[28]:


data.describe()


# In[29]:


#Check if this dataset contains missing values or not:
print(data.isnull().sum())


# In[8]:


#correlation between the features of this dataset:
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(15, 12))
sns.heatmap(data.corr())
plt.show()


# In[9]:


#visualize the data to analyze the unemployment rate. estimated number of employees according to different regions of India:
data.columns= ["States","Date","Frequency",
               "Estimated Unemployment Rate","Estimated Employed",
               "Estimated Labour Participation Rate","Region"]
plt.title("Indian Unemployment")
sns.histplot(x="Estimated Employed", hue="Region", data=data)
plt.show()
     


# In[10]:


#nemployment rate according to different regions of
plt.figure(figsize=(10, 8))
plt.title("Indian Unemployment")
sns.histplot(x="Estimated Unemployment Rate", hue="Region", data=data)
plt.show()


# In[30]:


#create a dashboard to analyze the unemployment rate of each Indian state by region
unemploment = data[["States", "Region", "Estimated Unemployment Rate"]]
figure = px.sunburst(unemploment, path=["Region", "States"], 
                     values="Estimated Unemployment Rate", 
                     width=700, height=700, color_continuous_scale="RdY1Gn", 
                     title="Unemployment Rate in India")
figure.show()


# In[12]:


sns.pairplot(data)
     


# In[13]:


data.describe()


# In[14]:


X = data[['Estimated Unemployment Rate', 'Estimated Employed', 'Estimated Labour Participation Rate']]

y = data['Estimated Employed']


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)

X_train


# In[17]:


from sklearn.linear_model import LinearRegression


# In[18]:


lm = LinearRegression()


# In[19]:


#fit the model inside it
lm.fit(X_train, y_train)


# In[20]:


#evaluating model
coeff_data = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])


# In[21]:


#This table is saying 
#if one unit is increase then area income will increase by $21
coeff_data


# In[22]:


#Predict the model
predictions = lm.predict(X_test)


# In[23]:


#plotting the prediction agains the target variable
plt.scatter(y_test, predictions)


# In[24]:


sns.distplot((y_test-predictions), bins=50);


# In[ ]:


#performed by adarsh pandey
#unemployment analysis and presdiction using machine learning 


# In[ ]:




