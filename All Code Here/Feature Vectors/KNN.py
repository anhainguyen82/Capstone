from keras.models import Sequential
from keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

plt.style.use('ggplot')
    
def clustering(data, cluster_number, title = None, visualize=False, labels=None):
    #convert dataframe to array
    if isinstance(data, pd.DataFrame):
        data = np.array(data)
       
    #fit the data    
    try:    
        kmeans = KMeans(n_clusters=cluster_number, random_state=2019).fit(data)
    except Exception:
        print("Data must be in an numpy array or pandas data frame")
    
    #create legend by dividing the number of job descriptions in each cluster by the total number
    #of scraped job description
    legend_label = []
    if labels is not None:
        labels = labels.copy()
        labels['C_label'] = kmeans.labels_
        labels['Counter'] = 1
        group_data = labels.groupby(['C_label','Label'])['Counter'].sum()
        group_data = group_data/group_data.groupby(level=1).sum()
        for index,new_df in group_data.groupby(level=0):
            legend_label.append(new_df.idxmax()[1])
        legend_label = pd.DataFrame(legend_label)
        #save legend to file
        #legend_label.to_csv('C:\\Users\\anhai\\Desktop\\SMU\\Capstone\\Neural Network\\legend.csv')
    
    #plot the data by transfroming the 512 dimensions vector with PCA
    if (visualize == True):
        # reduce word vector to 2D
        pca = PCA(n_components=2)
        reduced_vector = pca.fit_transform(data)
            
        # reduce the centroids to 2D
        centroid = pca.transform(kmeans.cluster_centers_)
        
        if title is not None:    
            fileName = "C:\\Users\\anhai\\Desktop\\SMU\\Capstone\\Neural Network\\{}_PCA_{}_Clusters.png".format(title, cluster_number)
        else:
            print("Must enter a title for plot to be saved.")
        
        fig = plt.figure()
        s = plt.scatter(reduced_vector[:,0], reduced_vector[:,1], c=kmeans.predict(data))
        if labels is not None:
            plt.legend(*s.legend_elements())
            plt.xlim(right=max(reduced_vector[:,0])+.3)
        plt.scatter(centroid[:, 0], centroid[:,1], s=150, c='r')
        plt.title('PCA reduced vectors')
        plt.xlabel('Component 1')
        plt.ylabel('component 2')
        #save plot to file
        #fig.savefig(fileName)
        
    #Sum of squared distances of samples to their closest cluster center.
    score = kmeans.inertia_ 
    
    return {'score': score, 'model':kmeans, 'legend':np.array(legend_label), 'labels':kmeans.labels_}


def plot_score(score, title):
    #plot score for number of clusters
    
    caption = "Sum of squared distances of samples to their closest cluster center."
    #set file name
    #fileName = "C:\\Users\\anhai\\Desktop\\SMU\\Capstone\\Neural Network\\{}_Inertia_Score.png".format(title)
    
    x = range(1, len(score) + 1)

    
    fig = plt.figure()
    plt.plot(x, score)
    plt.title('Score of number of clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squared Distance')
    fig.text(0,-.05,caption)
    #save plot to file
    #fig.savefig(fileName)
    
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

#sanity check
df_DA.shape
df_DB.shape
df_DE.shape
df_DS.shape
df_SE.shape
df_ST.shape

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
    
#sanity check
df_Combined.shape

find_number_of_clusters(df_Combined, "Combined", 20)

#clustering with labels
df_DA['Label'] = "DA"
df_DB['Label'] = "DB"
df_DE['Label'] = "DE"
df_DS['Label'] = "DS"
df_SE['Label'] = "SE"
df_ST['Label'] = "ST"

#sanity check
df_DA.shape
df_DB.shape
df_DE.shape
df_DS.shape
df_SE.shape
df_ST.shape

df_label = pd.DataFrame(pd.concat([df_DA['Label'], df_DB['Label'], 
                      df_DE['Label'], df_DS['Label'], 
                      df_SE['Label'], df_ST['Label']]))
#sanity check
df_label.shape

clustering(data=df_Combined, title="Combined", cluster_number=4,visualize=True)
full_dataset = clustering(data=df_Combined, title="Combined", cluster_number=6,visualize=True, labels=df_label)

#train/test split
x_train, x_test, y_train, y_test = train_test_split(df_Combined, df_label, test_size=0.25, random_state=2019)

predicted_labels=[]
for index, label in enumerate(full_dataset['labels']):
    predicted_labels.append(np.take(full_dataset['legend'], label))
    
#get train accuracy score of KNN prediction  
print("Accuracy:  {:.2f}%".format(accuracy_score(df_label, predicted_labels)*100))

#create KNN model object with training data
trained_KNN = clustering(data=x_train, labels=y_train, cluster_number=6)
#predict labels with x_test
test=trained_KNN['model'].predict(x_test)

#assign labels using legend provided by object
train_predicted_labels=[]
for index, label in enumerate(trained_KNN['labels']):
    train_predicted_labels.append(np.take(trained_KNN['legend'], label))
    
test_predicted_labels=[]
for index, label in enumerate(test):
    test_predicted_labels.append(np.take(trained_KNN['legend'], label))

#get train accuracy score of KNN prediction  
print("Testing Accuracy:  {:.2f}%".format(accuracy_score(y_train, train_predicted_labels)*100))
#get test accuracy score of KNN prediction    
print("Testing Accuracy:  {:.2f}%".format(accuracy_score(y_test, test_predicted_labels)*100))




