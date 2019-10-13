from keras.models import Sequential
from keras import layers
import pandas as pd
import numpy as np

#load Feature csv files
data_DA = pd.read_csv (r'C:\Users\anhai\Desktop\SMU\Capstone\Feature Vectors\DA_25_Feature_Vectors.csv', delimiter=' ')   
data_DB = pd.read_csv (r'C:\Users\anhai\Desktop\SMU\Capstone\Feature Vectors\DB_25_Feature_Vectors.csv', delimiter=' ')   
data_DE = pd.read_csv (r'C:\Users\anhai\Desktop\SMU\Capstone\Feature Vectors\DE_25_Feature_Vectors.csv', delimiter=' ')   
data_DS = pd.read_csv (r'C:\Users\anhai\Desktop\SMU\Capstone\Feature Vectors\DS_25_Feature_Vectors.csv', delimiter=' ')
data_SE = pd.read_csv (r'C:\Users\anhai\Desktop\SMU\Capstone\Feature Vectors\SE_25_Feature_Vectors.csv', delimiter=' ')   
data_ST = pd.read_csv (r'C:\Users\anhai\Desktop\SMU\Capstone\Feature Vectors\ST_25_Feature_Vectors.csv', delimiter=' ') 
     
#extract Feature_Vector column, add label columns, 0 for non-DS and 1 for DS
df_DA = pd.DataFrame(data_DA, columns= ['Feature_Vector'])
df_DA['Label'] = 0
df_DB = pd.DataFrame(data_DB, columns= ['Feature_Vector'])
df_DB['Label'] = 0
df_DE = pd.DataFrame(data_DE, columns= ['Feature_Vector'])
df_DE['Label'] = 0
df_DS = pd.DataFrame(data_DS, columns= ['Feature_Vector'])
df_DS['Label'] = 1
df_SE = pd.DataFrame(data_SE, columns= ['Feature_Vector'])
df_SE['Label'] = 0
df_ST = pd.DataFrame(data_ST, columns= ['Feature_Vector'])
df_ST['Label'] = 0

#concant all df into one
df = pd.concat([df_DA, df_DB, df_DE, df_DS, df_SE, df_ST])
df = df.reset_index(drop=True)
df = df.replace('\[\[','', regex=True)
df = df.replace(']]','', regex=True)
df = df.replace('\r\n','', regex=True)
df = df.replace('  ',',', regex=True)


model = Sequential()
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(np.array(df['Feature_Vector'].tolist()), np.array(df['Label']),
                    epochs=20,
                    verbose=False,
                    validation_data=(np.array(df['Feature_Vector'].tolist()), np.array(df['Label'])),
                    batch_size=10)
