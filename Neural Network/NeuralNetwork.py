from keras.models import Sequential
from keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

plt.style.use('ggplot')

def plot_history(history):
    #function written by 
    #N. Janakiev, "Practical Text Classification With Python and Keras", 
    #Real Python, 2018. [Online]. Available: https://realpython.com/python-keras-text-classification/. 
    #[Accessed: 14- Oct- 2019].
    #plot accuracy and validation loss history of neural network
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
def clustering(data, cluster_number, visualize=False):
    if isinstance(data, pd.DataFrame):
        data = np.array(df_DA)
    #elif not isinstance(data, np.array):
     #   print("Data must be in an numpy array or pandas data frame")
    try:    
        kmeans = KMeans(n_clusters=cluster_number).fit(data)
    except Exception:
        print("Data must be in an numpy array or pandas data frame")
        
    try:
        score = (silhouette_score(data, labels=kmeans.predict(data)))
    except Exception as error:
        print(error)
    
    if (visualize == True):
        # reduce word vector to 2D
        pca = PCA(n_components=2)
        reduced_vector = pca.fit_transform(data)
            
        # reduce the centroids to 2D
        centroid = pca.transform(kmeans.cluster_centers_)
        
        txt = "Plot of Component 1 and Component 2 reduced from a 512 features word vector using PCA"
        
        fig = plt.figure()
        plt.scatter(reduced_vector[:,0], reduced_vector[:,1], c=kmeans.predict(data))
        plt.scatter(centroid[:, 0], centroid[:,1], s=150, c='r')
        plt.title('Score of number of clusters')
        plt.xlabel('Component 1')
        plt.ylabel('component 2')
        fig.text(0,-.05,txt)
    
    return score
     
def plot_score(score):
    #plot score for number of clusters
    
    index = score.index(max(score))+2
    minimum = max(score)
    txt = "{} clusters has the minimum silhoutte score of {}".format(index, minimum)
    
    x = range(2, len(score) + 2)

    
    fig = plt.figure()
    plt.plot(x, score)
    plt.title('Score of number of clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhoutte Score')
    fig.text(0,-.05,txt)
    
def find_number_of_clusters(data, last, first=2):
    score = []

    for i in range(first,last):
        score.append(clustering(data,i))
    
    plot_score(score)


#load Feature csv files
df_DA = pd.read_csv('C:\\Users\\anhai\\Desktop\\SMU\\Capstone\\Feature Vectors\\dataAnalyst_FV.csv', header=None)
df_DB = pd.read_csv('C:\\Users\\anhai\\Desktop\\SMU\\Capstone\\Feature Vectors\\database_FV.csv', header=None)
df_DE = pd.read_csv('C:\\Users\\anhai\\Desktop\\SMU\\Capstone\\Feature Vectors\\dataEngineer_FV.csv', header=None)
df_DS = pd.read_csv('C:\\Users\\anhai\\Desktop\\SMU\\Capstone\\Feature Vectors\\dataScientist_FV.csv', header=None)
df_SE = pd.read_csv('C:\\Users\\anhai\\Desktop\\SMU\\Capstone\\Feature Vectors\\softwareEngineer_FV.csv', header=None)
df_ST = pd.read_csv('C:\\Users\\anhai\\Desktop\\SMU\\Capstone\\Feature Vectors\\statistician_FV.csv', header=None)

########## KNN ClUSTERING ##########
find_number_of_clusters(df_DA, 20)
find_number_of_clusters(df_DB, 20)
find_number_of_clusters(df_DE, 20)
find_number_of_clusters(df_DS, 20)
find_number_of_clusters(df_SE, 20)
find_number_of_clusters(df_ST, 20)

clustering(df_DS,2,True)

#convert dataframe into matrix array and concantonate into one
array = np.concatenate((np.array(df_DA), np.array(df_DB), np.array(df_DE), 
                        np.array(df_DS), np.array(df_SE), np.array(df_ST)), axis=0)

find_number_of_clusters(array, 20)

clustering(array,2,True)


########## NEURAL NETWORK ##########
#convert dataframe into matrix array and concantonate into one
array = np.concatenate((np.array(df_DA), np.array(df_DB), np.array(df_DE), 
                        np.array(df_DS), np.array(df_SE), np.array(df_ST)), axis=0)

array.shape #sanity check

#add labels, Data Scientist jobs are 1 all others 0
df_DA['Label'] = 0
df_DB['Label'] = 0
df_DE['Label'] = 0
df_DS['Label'] = 1
df_SE['Label'] = 0
df_ST['Label'] = 0

#convert dataframe into matrix array and concantonate into one
label = np.concatenate((np.array(df_DA['Label']), np.array(df_DB['Label']), np.array(df_DE['Label']), 
                        np.array(df_DS['Label']), np.array(df_SE['Label']), np.array(df_ST['Label'])), axis=0)

label.shape #sanity check

#train/test split
x_train, x_test, y_train, y_test = train_test_split(array, label, test_size=0.25, random_state=1000)

#neural network
model = Sequential()
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(x_train, y_train,
                    epochs=30,
                    verbose=False,
                    validation_data=(x_test, y_test),
                    batch_size=50)

plot_history(history)
model.summary()




