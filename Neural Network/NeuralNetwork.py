from keras.models import Sequential
from keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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
    
def clustering(data, title, cluster_number, visualize=False, labels=None):
    if isinstance(data, pd.DataFrame):
        data = np.array(data)
    #elif not isinstance(data, np.array):
     #   print("Data must be in an numpy array or pandas data frame")
    try:    
        kmeans = KMeans(n_clusters=cluster_number, random_state=2019).fit(data)
    except Exception:
        print("Data must be in an numpy array or pandas data frame")
    
    if (visualize == True):
        # reduce word vector to 2D
        pca = PCA(n_components=2)
        reduced_vector = pca.fit_transform(data)
            
        # reduce the centroids to 2D
        centroid = pca.transform(kmeans.cluster_centers_)
        
        fileName = "C:\\Users\\anhai\\Desktop\\SMU\\Capstone\\Neural Network\\{}_PCA_{}_Clusters.png".format(title, cluster_number)
        
        fig = plt.figure()
        s = plt.scatter(reduced_vector[:,0], reduced_vector[:,1], c=kmeans.predict(data))
        if labels is not None:
            labels['C_label'] = kmeans.labels_
            labels['Counter'] = 1
            group_data = labels.groupby(['C_label','Label'])['Counter'].sum()
            group_data = group_data/group_data.groupby(level=1).sum()
            legend_label = []
            for index,new_df in group_data.groupby(level=0):
                legend_label.append(new_df.idxmax()[1])
            legend_label = pd.DataFrame(legend_label)
            plt.legend(*s.legend_elements())
            plt.xlim(right=max(reduced_vector[:,0])+.3)
            legend_label.to_csv('C:\\Users\\anhai\\Desktop\\SMU\\Capstone\\Neural Network\\legend.csv')
        plt.scatter(centroid[:, 0], centroid[:,1], s=150, c='r')
        plt.title('PCA reduced vectors')
        plt.xlabel('Component 1')
        plt.ylabel('component 2')
        fig.savefig(fileName)
        
    #Sum of squared distances of samples to their closest cluster center.
    score = kmeans.inertia_ 
    
    return score


def plot_score(score, title):
    #plot score for number of clusters
    
    caption = "Sum of squared distances of samples to their closest cluster center."
    fileName = "C:\\Users\\anhai\\Desktop\\SMU\\Capstone\\Neural Network\\{}_Inertia_Score.png".format(title)
    
    x = range(1, len(score) + 1)

    
    fig = plt.figure()
    plt.plot(x, score)
    plt.title('Score of number of clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squared Distance')
    fig.text(0,-.05,caption)
    fig.savefig(fileName)
    
def find_number_of_clusters(data, title, last, first=1):
    score = []

    for i in range(first,last):
        score.append(clustering(data, title, i))
    
    plot_score(score, title)
    


#load Feature csv files
df_DA = pd.read_csv('C:\\Users\\anhai\\Desktop\\SMU\\Capstone\\Feature Vectors\\dataAnalyst_FV.csv', header=None)
df_DB = pd.read_csv('C:\\Users\\anhai\\Desktop\\SMU\\Capstone\\Feature Vectors\\database_FV.csv', header=None)
df_DE = pd.read_csv('C:\\Users\\anhai\\Desktop\\SMU\\Capstone\\Feature Vectors\\dataEngineer_FV.csv', header=None)
df_DS = pd.read_csv('C:\\Users\\anhai\\Desktop\\SMU\\Capstone\\Feature Vectors\\dataScientist_FV.csv', header=None)
df_SE = pd.read_csv('C:\\Users\\anhai\\Desktop\\SMU\\Capstone\\Feature Vectors\\softwareEngineer_FV.csv', header=None)
df_ST = pd.read_csv('C:\\Users\\anhai\\Desktop\\SMU\\Capstone\\Feature Vectors\\statistician_FV.csv', header=None)

########## KNN ClUSTERING ##########
find_number_of_clusters(data=df_DA, title="DA", last=20)
find_number_of_clusters(data=df_DB, title="DB", last=20)
find_number_of_clusters(data=df_DE, title="DE", last=20)
find_number_of_clusters(data=df_DS, title="DS", last=20)
find_number_of_clusters(data=df_SE, title="SE", last=20)
find_number_of_clusters(data=df_ST, title="ST", last=20)

clustering(data=df_DA, title="DA", cluster_number=1,visualize=True)
clustering(data=df_DB, title="DB", cluster_number=1,visualize=True)
clustering(data=df_DE, title="DE", cluster_number=1,visualize=True)
clustering(data=df_DS, title="DS", cluster_number=1,visualize=True)
clustering(data=df_SE, title="SE", cluster_number=1,visualize=True)
clustering(data=df_ST, title="ST", cluster_number=1,visualize=True)

#convert dataframe into matrix array and concantonate into one
df_Combined = pd.concat([df_DA, df_DB, 
                      df_DE, df_DS, 
                      df_SE, df_ST])

find_number_of_clusters(df_Combined, "Combined", 20)

#clustering with labels
df_DA['Label'] = "DA"
df_DB['Label'] = "DB"
df_DE['Label'] = "DE"
df_DS['Label'] = "DS"
df_SE['Label'] = "SE"
df_ST['Label'] = "ST"

df_label = pd.DataFrame(pd.concat([df_DA['Label'], df_DB['Label'], 
                      df_DE['Label'], df_DS['Label'], 
                      df_SE['Label'], df_ST['Label']]))

clustering(data=df_Combined, title="Combined", cluster_number=4,visualize=True)
clustering(data=df_Combined, title="Combined", cluster_number=6,visualize=True, labels=df_label)


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




