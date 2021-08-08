########################################################################
#       Customer Segmentation 
########################################################################


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plot some graphs, scatter plot in here 
import seaborn as sns # to beautify matplots
from sklearn.cluster import KMeans # for clustering data using KMeans modeling 

df = pd.read_csv(r'C:\Users\prakash.pandey\Downloads\Fashion_Segmentation.csv')

# Any results you write to the current directory are saved as output.
df.dropna(inplace=True)
print(df.head()) #printing 1st 5 rows of dataframe
print(df.describe()) # printing basic information about data like mean, median etc
print(df.info()) # printing information about metadata like dtypes, not null entries

sns.set() 
plt.style.use('ggplot')
# Performing basic Exploratory Data Analysis on important coloumns like Annual Income and Spending score
'''
pd.plotting.scatter_matrix(df[['Age(Med:33)', 'Spending Score(100)']], alpha = 0.6, figsize = [12,10], diagonal = 'hist', marker = 'o', color = '#66b3ff', hist_kwds={'color':['burlywood']})
plt.show()
plt.clf()
'''
#Modeling and evaluating cluster between Age and Spending Score
ds1 = df[['Age(Med:33)', 'Spending Score(100)']]
Ks = range(1,10)
inertia = []
for k in Ks:
    model = KMeans(n_clusters = k, random_state = 86)
    model.fit(ds1)
    inertia.append(model.inertia_)


# Ploting Inertia Plot to find best number of clusters
plt.plot(Ks, inertia, 'o')
plt.plot(Ks, inertia, '-', alpha = 0.45)
plt.ylabel('Inertia')
plt.xlabel('No. of Clusters')
plt.title('Interia plot for Age vs. Spending Score')
plt.show()
plt.clf()
# Number of clusters should be 3
model1 = KMeans(n_clusters = 3, random_state = 86)
model1.fit(ds1)
model1_pred = model1.predict(ds1)

labels1 = ['Young Shopaholic','Old Shoppers','Young Shoppers'] 
color1 = ['Blue','Yellow','Green']
d1=dict(zip(np.sort(np.unique(model1_pred)),labels1))
dc1=dict(zip(np.sort(np.unique(model1_pred)),color1))

# Converting dataframe into a numpy array
sc_x1 = np.array(ds1.iloc[:, 0])
sc_y1 = np.array(ds1.iloc[:, 1])
# Increasing the default figure size
plt.figure(figsize = (9,9))
for g in np.unique(model1_pred):
    ix = np.where(model1_pred == g)
    plt.scatter(sc_x1[ix], sc_y1[ix], c=dc1[g], s=50, cmap='viridis', label = d1[g])

# Adding centroids to the previous plot
plt.scatter(model1.cluster_centers_[:,0], model1.cluster_centers_[:,1], s=100, c='black', label = 'Centroid')
plt.ylabel('Spending Score (1-100)')
plt.xlabel('Age')
plt.title('Age vs. Spending Score')
plt.legend(loc='best')
plt.show()
plt.clf()
