from keras.models import Sequential
from keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from keras import optimizers

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
    
def clustering(data, cluster_number, title = None, visualize=False, labels=None):
    if isinstance(data, pd.DataFrame):
        data = np.array(data)
    #elif not isinstance(data, np.array):
     #   print("Data must be in an numpy array or pandas data frame")
    try:    
        kmeans = KMeans(n_clusters=cluster_number, random_state=2019).fit(data)
    except Exception:
        print("Data must be in an numpy array or pandas data frame")
        
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
        legend_label.to_csv('C:\\Users\\anhai\\Desktop\\SMU\\Capstone\\Neural Network\\legend.csv')
    
    # reduce word vector to 2D
    pca = PCA(n_components=2)
    reduced_vector = pca.fit_transform(data)
    prediction = kmeans.predict(data)
        
    if (visualize == True):
            
        # reduce the centroids to 2D
        centroid = pca.transform(kmeans.cluster_centers_)
        
        if title is not None:    
            fileName = "C:\\Users\\anhai\\Desktop\\SMU\\Capstone\\Neural Network\\{}_PCA_{}_Clusters.png".format(title, cluster_number)
        else:
            print("Must enter a title for plot to be saved.")
        
        fig = plt.figure()
        s = plt.scatter(reduced_vector[:,0], reduced_vector[:,1], c=prediction)
        if labels is not None:
            plt.legend(*s.legend_elements())
            plt.xlim(right=max(reduced_vector[:,0])+.3)
        plt.scatter(centroid[:, 0], centroid[:,1], s=150, c='r')
        plt.title('PCA reduced vectors')
        plt.xlabel('Component 1')
        plt.ylabel('component 2')
        fig.savefig(fileName)
        
    #Sum of squared distances of samples to their closest cluster center.
    score = kmeans.inertia_ 
    
    return {'score': score, 'model':kmeans, 'legend':np.array(legend_label), 'labels':kmeans.labels_,
            'x': reduced_vector[:,0], 'y':reduced_vector[:,1], 'c':prediction}


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

#label datasets with 1 for DS and 0 for non-DS
df_DA['Label'] = 0
df_DB['Label'] = 0
df_DE['Label'] = 0
df_DS['Label'] = 1
df_SE['Label'] = 0
df_ST['Label'] = 0

#sanity check
df_DA.shape
df_DB.shape
df_DE.shape
df_DS.shape
df_SE.shape
df_ST.shape

df_label_2 = pd.DataFrame(pd.concat([df_DA['Label'], df_DB['Label'], 
                      df_DE['Label'], df_DS['Label'], 
                      df_SE['Label'], df_ST['Label']]))
#sanity check
df_label_2.shape

#cluster
full_dataset_2 = clustering(data=df_Combined, title="Combined_2Clusters", cluster_number=2,visualize=True, labels=df_label_2)

#find accuracy score
predicted_labels=[]
for index, label in enumerate(full_dataset_2['labels']):
    predicted_labels.append(np.take(full_dataset_2['legend'], label))
    
#get train accuracy score of KNN prediction  
print("Accuracy:  {:.2f}%".format(accuracy_score(df_label_2, predicted_labels)*100))

#label dataset with job titles
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

df_label_6 = pd.DataFrame(pd.concat([df_DA['Label'], df_DB['Label'], 
                      df_DE['Label'], df_DS['Label'], 
                      df_SE['Label'], df_ST['Label']]))
#sanity check
df_label_6.shape

#cluster
clustering(data=df_Combined, title="Combined", cluster_number=4,visualize=True)
full_dataset = clustering(data=df_Combined, title="Combined", cluster_number=6,visualize=True, labels=df_label_6)

#find accuracy score
predicted_labels=[]
for index, label in enumerate(full_dataset['labels']):
    predicted_labels.append(np.take(full_dataset['legend'], label))
    
#get train accuracy score of KNN prediction  
print("Accuracy:  {:.2f}%".format(accuracy_score(df_label_6, predicted_labels)*100))

#train/test split
x_train, x_test, y_train, y_test = train_test_split(df_Combined, df_label_6, test_size=0.25, random_state=2019)

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

    
########## NEURAL NETWORK ##########
#sanity check
df_Combined.shape
df_label_2.shape

#PCA transformation of df_Combined
pca = PCA(n_components=2)
reduced_vector = pca.fit_transform(df_Combined)

#train/test split
x_train, x_test, y_train, y_test = train_test_split(df_Combined, 
                                                    df_label_2, 
                                                    test_size=0.25, 
                                                    random_state=2019)

#neural network
#tunable: unit in first layer, additional layers, dropout rates inbetween layers
#optimizer learning rate, batch size 
model = Sequential()
model.add(layers.Dense(256, input_dim = 512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid')) #sigmoid for binary class
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy', #cross entropy for binary class
              metrics=['accuracy'])

history = model.fit(np.array(x_train), np.array(y_train),
                    epochs=30,
                    verbose=False,
                    validation_data=(np.array(x_test), np.array(y_test)),
                    batch_size=75)

plot_history(history)

#model remains at 90% accuracy (levels off at ~15 epochs), not able to find parameters to increase
model = Sequential()
model.add(layers.Dense(256, input_dim = 512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(np.array(df_Combined), np.array(df_label_2),
                    epochs=15,
                    verbose=False,
                    batch_size=75)

results_2 = model.predict_classes(np.array(df_Combined))
model.summary()

#get train accuracy score of NN prediction  
print("Accuracy:  {:.2f}%".format(accuracy_score(df_label_2, results_2)*100))

#visualize results
fig = plt.figure()
s = plt.scatter(reduced_vector[:,0], reduced_vector[:,1], c=results_2[:,0])
plt.legend(*s.legend_elements())
plt.xlim(right=max(reduced_vector[:,0])+.3)
plt.title('PCA reduced vectors')
plt.xlabel('Component 1')
plt.ylabel('component 2')
fig.savefig("C:\\Users\\anhai\\Desktop\\SMU\\Capstone\\Neural Network\\NN_2_clusters.png")

#categorize labels
df_label_6 = pd.Categorical(df_label_6['Label'])

#train/test split
x_train, x_test, y_train, y_test = train_test_split(df_Combined, 
                                                    df_label_6.codes, 
                                                    test_size=0.25, 
                                                    random_state=2019)
#neural network
#tunable: unit in first layer, additional layers, dropout rates inbetween layers
#optimizer learning rate, batch size 
model = Sequential()
model.add(layers.Dense(512, input_dim = 512, activation='relu'))
model.add(layers.Dense(6, activation='softmax')) #softmax for multiple classes
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy', #sparse_categorical_crossentropy for multiple classes, intergers as labels
              metrics=['accuracy'])

history = model.fit(np.array(x_train), np.array(y_train),
                    epochs=30,
                    verbose=False,
                    validation_data=(np.array(x_test), np.array(y_test)),
                    batch_size=75)

plot_history(history)

model = Sequential()
model.add(layers.Dense(512, input_dim = 512, activation='relu'))
model.add(layers.Dense(6, activation='softmax')) #softmax for multiple classes
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy', #sparse_categorical_crossentropy for multiple classes, intergers as labels
              metrics=['accuracy'])

model.fit(np.array(df_Combined), np.array(df_label_6.codes),
                    epochs=15,
                    verbose=False,
                    batch_size=75)

results_6 = model.predict_classes(np.array(df_Combined))
model.summary()

#get train accuracy score of NN prediction  
print("Accuracy:  {:.2f}%".format(accuracy_score(df_label_6.codes, results_6)*100))

#visualize results
fig = plt.figure()
s = plt.scatter(reduced_vector[:,0], reduced_vector[:,1], c=results_6)
plt.legend(*s.legend_elements())
plt.xlim(right=max(reduced_vector[:,0])+.3)
plt.title('PCA reduced vectors')
plt.xlabel('Component 1')
plt.ylabel('component 2')
fig.savefig("C:\\Users\\anhai\\Desktop\\SMU\\Capstone\\Neural Network\\NN_6_clusters.png")



