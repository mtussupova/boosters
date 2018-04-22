
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


company = pd.read_excel('компании.xlsx', encoding="windows-1251")
products = pd.read_excel('продукты.xlsx', encoding="windows-1251")
acc = pd.read_excel('счета.xlsx', encoding="windows-1251")
tarif = pd.read_excel('тарифы.xlsx', encoding="windows-1251")


# Data Preproccesing

# In[3]:


acc = acc.rename(columns={'ID_CLIENT': 'ID_Client'})
company = company.rename(columns={'ID_Client': 'ID_Client'})


# In[4]:


df_tarif_acc = pd.merge(tarif, acc, on='ID_Client')


# In[5]:


df_tarif_acc['Закрыт'] = df_tarif_acc['Закрыт'].replace(to_replace=['Y', 'N'], value=[1, 0])


# In[6]:


df_tarif_acc['Статус'] = df_tarif_acc['Статус'].replace(to_replace=['Работает', 'Закрыт'], value=[1, 0])


# In[7]:


df_tarif_acc['Филиал'] = df_tarif_acc['Филиал'].map(lambda x: str(x)[18:])


# In[8]:


df_1 = pd.merge(df_tarif_acc, company, on='ID_Client')


# In[9]:


df_1['TypeClient'] = df_1['TypeClient'].replace(to_replace=['ИП', 'ЮЛ'], value=[1, 2])


# In[10]:


products = products.assign(pr1=0)


# In[11]:


products['pr1'] = (products['Продукт']== 1)


# In[12]:


products = products.rename(columns={'ID_CLIENT': 'ID_Client'})


# In[13]:


df_p = products.drop(['Продукт'], axis=1)


# In[14]:


df = pd.merge(df_1, df_p, on=['ID_Client'])


# Splitting data into test and train datasets

# In[15]:


y = df['pr1']
X = df.drop(['pr1'], axis = 1)


# Changing all missing data to 0

# In[16]:


X = X.fillna(0)


# We have dropped ID_Client and ID_ACC in order to SVM to work

# In[17]:


X = X.drop(['ID_Client', 'ID_ACC'], axis = 1)


# In[19]:


X['Статус'] = X['Статус'].replace(to_replace=['Предварительный ввод', 'Ожидает подтверждения'], value=[2, 3])


# In[20]:


from sklearn.model_selection  import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.33, random_state=5)


# Implementing SVM to dataset

# In[ ]:


from sklearn import svm
clf = svm.SVC()
clf.fit(X_train, Y_train)


# In[ ]:


pred = clf.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score
print("Accuracy score:", accuracy_score(Y_test, pred))

