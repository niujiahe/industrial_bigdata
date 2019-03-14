
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import pywt
from math import log,asinh,atan


# In[2]:


os_dir1 = "/home/njh/data/origin/Train/02/Sensor/"
os_dir2 = "/home/njh/data/origin/TrainNew/01-TrainingData-additional/02/Sensor/"
#os_dir2 = "/home/njh/data/origin/TestNew/05/Sensor/"
app_train_plc = pd.read_csv("/home/njh/data/origin/Train/02/PLC/plc.csv")


file_name_all = os.listdir(os_dir1)
print(file_name_all)


# In[3]:


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: 
        currentLabel = featVec
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1 
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries 
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# In[4]:


app_train_merge_sensor= pd.DataFrame()
for name in range(0,len(file_name_all)):
#for name in range(0,1):
    app_train_merge_sensor_m= pd.DataFrame()
    print(str(name+1) + " Begin...")
    #origin data
    #app_train1 = pd.read_csv(os_dir1+str(name+1)+'.csv')
    #app_train1.rename(columns={'vibration_1':'current_1','vibration_2':'current_2','vibration_3':'current_3','current':'vibration_1'}, inplace=True) 
    #new data
    app_train = pd.read_csv(os_dir2+str(name+1)+'.csv')
    #app_train2.drop(['vibration_1'],axis=1,inplace = True)
    app_train.drop(['current'],axis=1,inplace = True)
    
    #app_train = app_train1.join(app_train2)
    
    app_train[app_train>100] = 100 
    app_train[app_train<-100] = -100 
    app_train = app_train[~app_train.isin([100,-100,np.nan, np.inf, -np.inf]).any(1)]
    app_train = app_train.reset_index(drop=True)
    print(str(name+1)+'.csv'+"merge Done\n")
    print('merge shape: ', app_train.shape)
    
    #提取统计量
    plus_cells = int(app_train.shape[0]/600)
    for k in range(0,600):
        app_train_merge_sensor_s= pd.DataFrame()
        for wpd_temp in range(0,6):
            app_train_temp_origin=app_train[plus_cells*k:plus_cells*(k+1)-1]
            app_train_temp_origin = app_train_temp_origin.reset_index(drop=True)

            app_train_temp = pd.DataFrame()
            for col in app_train_temp_origin.columns:
                x = app_train_temp_origin[col]
                cA, cD = pywt.dwt(x,'sym2')
                for wpd_time in range(0,wpd_temp):
                    cA, cD = pywt.dwt(cA,'sym2')
                cD_df = pd.DataFrame(cD, columns = [col+'_cD'+str(wpd_temp)])
                app_train_temp = pd.concat([app_train_temp,cD_df],axis =1)

            #方差
            var_temp = app_train_temp.var()
            app_train_merge_sensor_s = pd.concat([app_train_merge_sensor_s,pd.DataFrame([var_temp.values], columns = var_temp.index+'_var')],axis =1)
            #均值
            mean_temp = app_train_temp.mean()
            app_train_merge_sensor_s = pd.concat([app_train_merge_sensor_s,pd.DataFrame([mean_temp.values], columns = mean_temp.index+'_mean')],axis =1)
            #中位数
            median_temp = app_train_temp.median()
            app_train_merge_sensor_s = pd.concat([app_train_merge_sensor_s,pd.DataFrame([median_temp.values], columns = median_temp.index+'_median')],axis =1)
            #Energy
            temp_Energy = []
            for col in app_train_temp.columns:
                x = app_train_temp[col]
                x_energy = float('%.8f' % sum([pow(i,2) for i in x]))
                #print(x_energy)
                temp_Energy.append(x_energy) 
            energy_df = pd.DataFrame([temp_Energy], columns = app_train_temp.columns+'_energy')
            app_train_merge_sensor_s = pd.concat([app_train_merge_sensor_s,energy_df],axis =1)
            #RMS
            temp_rms = []
            for col in app_train_temp.columns:
                x = app_train_temp[col]
                energy = float(app_train_merge_sensor_s[col+'_energy'])
                temp_rms.append((energy/plus_cells)**0.5) 
            rms_df = pd.DataFrame([temp_rms], columns = app_train_temp.columns+'_rms')
            app_train_merge_sensor_s = pd.concat([app_train_merge_sensor_s,rms_df],axis =1)
            #Crest
            temp_Crest = []
            for col in app_train_temp.columns:
                x = app_train_temp[col]
                rms = float(app_train_merge_sensor_s[col+'_rms'])
                temp_Crest.append(x.max()/rms) 
            Crest_df = pd.DataFrame([temp_Crest], columns = app_train_temp.columns+'_Crest')
            app_train_merge_sensor_s = pd.concat([app_train_merge_sensor_s,Crest_df],axis=1)
            #Kurtosis
            temp_Kurtosis = []
            for col in app_train_temp.columns:
                x = app_train_temp[col]
                x_mean = x.mean()
                x_var = float('%.8f' %x.var())
                if(x_var == 0):
                    x_var = 1
                x_pow4 = float('%.4f' % sum([pow(i-x_mean,4) for i in x]))
                temp_Kurtosis.append(x_pow4/(plus_cells - 1) / x_var / x_var) 
            Kurtosis_df = pd.DataFrame([temp_Kurtosis], columns = app_train_temp.columns+'_Kurtosis')
            app_train_merge_sensor_s = pd.concat([app_train_merge_sensor_s,Kurtosis_df],axis =1)
            #Line
            temp_Line = []
            for col in app_train_temp.columns:
                x = app_train_temp[col]
                sum1=0
                for i in range(0,len(x)-1):
                    sum1 = sum1 + abs(x[i+1]-x[i])
                temp_Line.append(sum1) 
            Line_df = pd.DataFrame([temp_Line], columns = app_train_temp.columns+'_Line')
            app_train_merge_sensor_s = pd.concat([app_train_merge_sensor_s,Line_df],axis =1)
            #Entropy
            temp_Entropy = []
            for col in app_train_temp.columns:
                x = app_train_temp[col]
                k=500
                d1 = pd.cut(x,k,labels = range(k))
                Entropy1 = calcShannonEnt(d1)
                temp_Entropy.append(Entropy1)
            Entropy_df = pd.DataFrame([temp_Entropy], columns = app_train_temp.columns+'_Entropy')
            app_train_merge_sensor_s = pd.concat([app_train_merge_sensor_s,Entropy_df],axis =1)

            #asinh
            temp_asinh = []
            for col in app_train_temp.columns:
                x_col = app_train_temp[col]
                temp = np.array([asinh(x) for x in x_col])
                temp_asinh.append(temp.std())
            asinh_df = pd.DataFrame([temp_asinh], columns = app_train_temp.columns+'_asinh')
            app_train_merge_sensor_s = pd.concat([app_train_merge_sensor_s,asinh_df],axis =1)

            #atan
            temp_atan = []
            for col in app_train_temp.columns:
                x_col = app_train_temp[col]
                temp = np.array([atan(x) for x in x_col])
                temp_atan.append(temp.std())
            atan_df = pd.DataFrame([temp_atan], columns = app_train_temp.columns+'_atanh')
            app_train_merge_sensor_s = pd.concat([app_train_merge_sensor_s,atan_df],axis =1)
            
            
        
        #print('second shape: ', app_train_merge_sensor_s.shape)
        app_train_merge_sensor_m = pd.concat([app_train_merge_sensor_m,app_train_merge_sensor_s], ignore_index = True)
    
    
    app_train_merge_sensor_m['RUL']  = 237.5 - 5 * name
    print('minute shape: ', app_train_merge_sensor_m.shape)
    
    app_train_merge_sensor = pd.concat([app_train_merge_sensor,app_train_merge_sensor_m], ignore_index = True)
    print(str(name+1)+'.csv'+"concat Done")
    print('concat shape: ', app_train_merge_sensor.shape)
    print("-----------------------------------------------------------------")


# In[5]:


#app_train_merge_sensor = app_train_merge_sensor.dropna(axis=0,how='any')


# In[6]:


app_train_merge_sensor.to_csv('/home/njh/final_data/'+'Train_WPD02.csv',index='False')


# In[7]:


app_train_merge_sensor.describe()

