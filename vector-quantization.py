
# coding: utf-8

# In[19]:


import numpy as np
from os import listdir, getcwd
from os.path import isfile, join
import random


# In[2]:


def get_and_save_train_data(datapath, output_file):
    '''Gets train instances and save '''
    commands_dirs_by_readers = [join(datapath,f) for f in listdir(datapath) if not isfile(join(datapath, f))]
    instances = np.concatenate([np.concatenate([get_mfc_file(join(join(directory, 'komutlar'), f))]) for directory in commands_dirs_by_readers for f in listdir(join(directory, 'komutlar')) if f.endswith('.mfc')])
    np.save(output_file, instances)
    return instances


# In[3]:


def load_datapoints(filepath):
    return np.load(filepath)


# In[4]:


def get_mfc_file(path):
    '''Loads mfc file as matrix'''
    return np.loadtxt(path)


# In[5]:


def get_and_save_test_data(datapath, output_file):
    '''Gets and saves test instances'''
    test_instances = np.concatenate([np.concatenate([get_mfc_file(join(datapath,f))]) for f in listdir(datapath) if f.endswith('.mfc')])
    np.save(output_file, test_instances)
    return test_instances


# In[6]:


train_data = get_and_save_train_data('../assignment-1/ProjectData/TrainData', 'train-datapoints')
test_data = get_and_save_test_data('../assignment-1/ProjectData/EvalData', 'test-datapoints')

# In[7]:


cluster_schedule = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64]


# In[30]:


def distance(point1, point2):
    return np.asscalar(np.linalg.norm(point1-point2, ord=2))


# In[9]:


def cluster_distortion(cluster, centroid):
    return sum([distance(point, centroid) for point in cluster])


# In[12]:


def total_distortion(clusters):
    return sum([cluster_distortion(points, centroid) for centroid, points in clusters])


# In[47]:
def find_cluster(point, centroids):
    dist_to_centroids = []   
    for centroid in centroids:
        dist = distance(point, centroid)
        dist_to_centroids.append(dist)
    return [i for i in sorted(enumerate(dist_to_centroids), key=lambda x:x[1])][0]

def kmeans_clustering(points, number_of_clusters):
    centroid_indices = random.sample(range(len(points)), number_of_clusters)
    centroids = [points[index] for index in centroid_indices]
    clusters = [[] for i in range(number_of_clusters)]
    terminate = False
    last_distortion = total_distortion([(np.matrix(points).mean(0), points)])  
    #print('initial')
    #print(centroids)         
    while terminate == False:
        clusters = [[] for i in range(number_of_clusters)]
        for point in points:
            dist_to_centroids = [distance(point, centroid) for centroid in centroids]
            cluster_index = [i[0] for i in sorted(enumerate(dist_to_centroids), key=lambda x:x[1])][0]    
            clusters[cluster_index].append(point)
        centroids = [np.matrix(points).mean(0) for points in clusters]     
        #print('next')
        #print(centroids)        
        distortion = total_distortion(zip(centroids, clusters))
        if last_distortion - distortion < 1.0:
            terminate = True
        else:
            print('Last distortion: ' + str(last_distortion) + ' New distortion: '  + str(distortion))
            last_distortion = distortion  
    return [(centroid, points) for centroid, points in zip(centroids, clusters)]           


# In[48]:


clusters = []
split_type = 'binary-recursive'
train_distortions = []
test_distortions = []
for number_of_cluster in cluster_schedule:
    print('Schedule ' + str(number_of_cluster))
    if number_of_cluster == 1:
        clusters = [(train_data.mean(0), train_data)]
    else:
        if split_type == 'binary-recursive':
            cluster_to_split = number_of_cluster - len(clusters)
            for iteration in range(cluster_to_split):
                distortions = []
                for centroid, cluster_points in clusters:
                    distortions.append(cluster_distortion(cluster_points, centroid))
                cluster_index = [i[0] for i in sorted(enumerate(distortions), key=lambda x:x[1], reverse=True)][0]    
                centroid, datapoints = clusters[cluster_index]
                clusters = clusters + kmeans_clustering(datapoints, 2)
                del clusters[cluster_index]
        elif split_type == 'binary':
            # choose number_of_cluster - len(clusters) and split each of them into two clusters 
            cluster_to_split = number_of_cluster - len(clusters)
            distortions = []
            for centroid, cluster_points in clusters:
                distortions.append(cluster_distortion(cluster_points, centroid))
            cluster_indices_to_split = [i[0] for i in sorted(enumerate(distortions), key=lambda x:x[1], reverse=True)][:cluster_to_split]    
            for cluster_index in cluster_indices_to_split:
                centroid, datapoints = clusters[cluster_index]
                clusters = clusters + kmeans_clustering(datapoints, 2)
            for cluster_index in cluster_indices_to_split:
                del clusters[cluster_index]
                
        elif split_type == 'multiple': 
            # choose one cluster and split into number_of_cluster - len(clusters) clusters
            cluster_to_split = number_of_cluster - len(clusters)
            distortions = []
            for centroid, cluster_points in clusters:
                distortions.append(cluster_distortion(cluster_points, centroid))
            cluster_index = [i[0] for i in sorted(enumerate(distortions), key=lambda x:x[1], reverse=True)][0]    
            centroid, datapoints = clusters[cluster_index]
            clusters = clusters + kmeans_clustering(datapoints, cluster_to_split)
            del clusters[cluster_index]
        else:
            print('Not a valid split type')    
    #print(clusters[0])  
    train_distortion = total_distortion(clusters)          
    train_distortions.append(train_distortion)
    test_distortion = 0
    for point in test_data:
        cluster_index, dist = find_cluster(point, [centroid for centroid, points in clusters])
        test_distortion += dist
    test_distortions.append(test_distortion)
    
    print('Train distortion ' + str(train_distortion))
    print('Test distortion ' + str(test_distortion))    
print(train_distortions)    
print(test_distortions)  
