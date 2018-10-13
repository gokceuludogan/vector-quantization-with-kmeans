# coding: utf-8

import numpy as np
from os import listdir, getcwd
from os.path import isfile, join
import random
import argparse
import matplotlib.pyplot as plt

class VectorQuantizer:

    cluster_schedule = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64]

    def __init__(self, train_data, test_data, distance_method, centroid_method, cluster_method, split_type, output):
        self.train_data = train_data
        self.test_data = test_data
        self.distance_method = distance_method
        self.centroid_method = centroid_method
        self.cluster_method = cluster_method
        self.split_type = split_type
        self.output = output
        print('Vector Quantizer is initialized with following parameters')
        print('distance ' + str(distance_method))
        print('centroid method ' + str(centroid_method))
        print('split type ' + str(split_type))

    def distance(self, point1, point2):
        return np.linalg.norm(point1-point2, ord=self.distance_method)

    def cluster_distortion(self, cluster, centroid):
        return sum([self.distance(point, centroid) for point in cluster])

    def total_distortion(self, clusters):
        return sum([self.cluster_distortion(points, centroid) for centroid, points in clusters])

    def find_cluster(self, point, clusters):
        if self.cluster_method == 'centroid':
            distances = [self.distance(point, centroid) for centroid, points in clusters]
        elif self.cluster_method == 'point':
            distances = [min(self.distance(point, cluster_point) for cluster_point in points) for centroid, points in clusters]
        elif self.cluster_method == 'average':
            distances = []   
            for centroid, points in clusters:           
                distances_to_cluster = [self.distance(point, cluster_point) for cluster_point in points]
                distances.append(sum(distances_to_cluster)/ float(len(distances_to_cluster)))
        else: 
            raise ValueError('{cluster_method} wrong, use "centroid", "point" or "average"'.format(cluster_method=repr(self.cluster_method)))
        
        return [i for i in sorted(enumerate(distances), key=lambda x:x[1])][0]     

    def calc_centroids(self, points):
        if self.centroid_method == 'mean':
            return np.matrix(points).mean(0)
        else:
        #elif self.centroid_method == 'median':
            return np.median(points, axis=0)
        #else:
        #    raise ValueError('{centroid_method} wrong, use "mean" or "median"'.format(centroid_method=repr(centroid_method)))
    def kmeans_clustering(self, points, number_of_clusters):
        print('KMeans')
        centroid_indices = random.sample(range(len(points)), number_of_clusters)
        centroids = [points[index] for index in centroid_indices]
        clusters = [[] for i in range(number_of_clusters)]
        terminate = False
        last_distortion = self.total_distortion([(self.calc_centroids(points), points)])       
        while terminate == False:
            clusters = [[] for i in range(number_of_clusters)]
            for point in points:
                cluster_index, dist = self.find_cluster(point, zip(centroids, clusters)) 
                clusters[cluster_index].append(point)
            centroids = [self.calc_centroids(points) for points in clusters]          
            distortion = self.total_distortion(zip(centroids, clusters))
            print('Last distortion: ' + str(last_distortion) + ' New distortion: '  + str(distortion))
            if last_distortion == distortion:
                terminate = True
            else:
                last_distortion = distortion  
        return [(centroid, points) for centroid, points in zip(centroids, clusters)]           
    
    def quantize(self):
        datasizes = {'train': len(self.train_data), 'test': len(self.test_data)}
        clusters = []
        train_distortions = []
        test_distortions = []
        
        for number_of_cluster in self.cluster_schedule:
            print('Schedule ' + str(number_of_cluster))
            if number_of_cluster == 1:
                clusters = [(self.calc_centroids(self.train_data), self.train_data)]
            else:
                if self.split_type == 'binary-recursive':
                    split_size = 2
                    cluster_to_split = 1
                    number_of_iteration = number_of_cluster - len(clusters)

                elif self.split_type == 'binary':
                    split_size = 2 
                    # choose number_of_cluster - len(clusters) and split each of them into two clusters 
                    cluster_to_split = number_of_cluster - len(clusters)
                    number_of_iteration = 1
                elif self.split_type == 'multiple': 
                    # choose one cluster and split into number_of_cluster - len(clusters) clusters
                    split_size = number_of_cluster - len(clusters) + 1 
                    cluster_to_split = 1
                    number_of_iteration = 1
                else:
                    raise ValueError('{split_type} wrong, use "binary", "binary-recursive" or "multiple"'.format(split_type=repr(self.split_type)))
                for iteration in range(number_of_iteration):
                    distortions = []
                    for centroid, cluster_points in clusters:
                        distortions.append(self.cluster_distortion(cluster_points, centroid))
                    cluster_indices_to_split = [i[0] for i in sorted(enumerate(distortions), key=lambda x:x[1], reverse=True)][:cluster_to_split]    
                    for cluster_index in cluster_indices_to_split:
                        centroid, datapoints = clusters[cluster_index]
                        clusters = clusters + self.kmeans_clustering(datapoints, split_size)
                    for cluster_index in cluster_indices_to_split:
                        del clusters[cluster_index]

            train_distortion = self.total_distortion(clusters)          
            train_distortions.append(train_distortion)
            test_distortion = sum([self.find_cluster(point, clusters)[1] for point in self.test_data])
            test_distortions.append(test_distortion)
            print('Train distortion ' + str(train_distortion))
            print('Test distortion ' + str(test_distortion))  

        self.save_distortions(train_distortions, test_distortions)
        self.plot_distortions(train_distortions, test_distortions)                  

    def plot_distortions(self, train_distortions, test_distortions):
        plt.figure(1)
        plt.plot(self.cluster_schedule, train_distortions, label='Training distortions')
        plt.plot(self.cluster_schedule, test_distortions, label='Test distortions')
        plt.xlabel("Schedule")
        plt.ylabel("Total Distortions")
        plt.savefig(self.split_type + 'total_distortions.png')
        plt.figure(2)
        plt.plot(self.cluster_schedule, [distortion/len(self.train_data) for distortion in train_distortions], label='Training distortions per point')
        plt.plot(self.cluster_schedule, [distortion/len(self.test_data) for distortion in test_distortions], label='Test distortions per point')
        plt.xlabel("Schedule")
        plt.ylabel("Distortions per point")
        plt.savefig(self.split_type + 'distortions_per_point.png')

    def save_distortions(self, train_distortions, test_distortions):
        output = open(self.split_type + self.output, 'w')
        output.write('Train:' + ','.join(str(d) for d in train_distortions) + '\n')
        output.write('Test:' + ','.join(str(d) for d in test_distortions) + '\n')
        output.write('TrainPerInstance:' + ','.join(str(dist/len(self.train_data)) for dist in train_distortions) + '\n')
        output.write('TestPerInstance:' + ','.join(str(dist/len(self.test_data)) for dist in test_distortions) + '\n')
        output.close()




def get_and_save_train_data(datapath, output_file):
    '''Gets train instances and save '''
    commands_dirs_by_readers = [join(datapath,f) for f in listdir(datapath) if not isfile(join(datapath, f))]
    instances = np.concatenate([np.concatenate([get_mfc_file(join(join(directory, 'komutlar'), f))]) for directory in commands_dirs_by_readers for f in listdir(join(directory, 'komutlar')) if f.endswith('.mfc')])
    np.save(output_file, instances)
    return instances

def load_datapoints(filepath):
    return np.load(filepath)


def get_mfc_file(path):
    '''Loads mfc file as matrix'''
    return np.loadtxt(path)


def get_and_save_test_data(datapath, output_file):
    '''Gets and saves test instances'''
    test_instances = np.concatenate([np.concatenate([get_mfc_file(join(datapath,f))]) for f in listdir(datapath) if f.endswith('.mfc')])
    np.save(output_file, test_instances)
    return test_instances


def main(args):
    if args.load:
        train_data = load_datapoints(args.trainpoints)
        test_data = load_datapoints(args.testpoints)
    else:
        train_data = get_and_save_train_data(args.train, 'train-datapoints')
        test_data = get_and_save_test_data(args.test, 'test-datapoints')
    if args.distance == 'l2':
        vq = VectorQuantizer(train_data, test_data, 2, 'mean', 'centroid', args.splittype, args.output)
    elif args.distance == 'l1':
        vq = VectorQuantizer(train_data, test_data, 1, 'median', 'centroid', args.splittype, args.output)
    else: 
        raise ValueError('{distance} wrong, use "l2" or "l1"'.format(distance=repr(args.distance)))

    vq.quantize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', default=False)
    parser.add_argument('--trainpoints', default='train-datapoints.npy')
    parser.add_argument('--testpoints', default='test-datapoints.npy')
    parser.add_argument("--train", default=join(join('../assignment-1', 'ProjectData'), 'TrainData'))
    parser.add_argument("--test", default=join(join('../assignment-1', 'ProjectData'), 'EvalData'))
    parser.add_argument("--splittype", default='binary-recursive')
    parser.add_argument("--distance", default='l2')
    parser.add_argument("--output", default='distortions.log')
    args = parser.parse_args()
    main(args)