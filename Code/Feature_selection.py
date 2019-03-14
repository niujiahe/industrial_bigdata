
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math


# In[2]:


def unit_f(z):
    if (z>=0):
        ans = 1
    else:
        ans = -1
    return ans


# In[3]:


def unit_f2(z):
    ans = 0
    if (z>0):
        ans = 1
    elif (z<0):
        ans = -1
    return ans


# In[4]:


def Pearson_correlation(x,v):
    x_var = np.std(x)
    v_var = np.std(v)
    xv_cov= np.cov(x,v)
    return xv_cov/x_var/v_var


# In[5]:


def monotonicity(x):
    dx1 = 0
    dx2 = 0
    for i in range (0,len(x)-1):
        dx1 = dx1 + unit_f(x[i+1] - x[i])
        dx2 = dx2 + unit_f(x[i] - x[i+1])
    return abs(dx1 - dx2)/(len(x)-1)


# In[6]:


def monotonicity2(x):
    dx1 = 0
    for i in range (0,len(x)-1):
        dx1 = dx1 + unit_f2(x[i+1] - x[i])
    return dx1/(len(x)-1)


# In[7]:


def autocorrelation(x):
    ans = 0
    for i in range(0,len(x)-1):
        ans = ans + pow((x[i+1] - x[i]),2)
    return ans/(len(x)-1)


# In[8]:


Train1 = pd.read_csv('/home/njh/final_data/Test_DWT01_Test3.csv',index_col=0)
Train2 = pd.read_csv('/home/njh/final_data/PHM2012_HHT01.csv',index_col=0)


# In[9]:


Train1 = Train1.dropna(axis=0,how='any')
Train1 = Train1.reset_index(drop=True)
Train2 = Train2.dropna(axis=0,how='any')
Train2 = Train2.reset_index(drop=True)


# In[27]:


for col in Train2.columns:
    print(col)
    print(autocorrelation(Train2[col]))


# In[19]:


len(Train2['imf0v2_rms'])


# In[23]:


from scipy.signal import hilbert, chirp
analytic_signal = hilbert(Train2['imf0v2_rms'])
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) )


# In[24]:


monotonicity2(instantaneous_frequency)


# In[12]:


df2 = pd.DataFrame(np.random.randint(low=0, high=10, size=(5, 5)),columns=['a', 'b', 'c', 'd', 'e'])


# In[13]:


mean_df = df2.mean()


# In[14]:


df2


# In[15]:


app_train_merge_sensor_s = pd.DataFrame([mean_df.values], columns = mean_df.index+'_var')
        


# In[16]:


app_train_merge_sensor_s


# In[17]:


mean_df

