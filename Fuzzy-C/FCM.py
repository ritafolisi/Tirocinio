import copy
import math
import random
import time
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import decimal
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error

class FCM():
    def __init__(self, MAX=10000.0, Epsilon=0.00000001):
        self.MAX = MAX
        self.Epsilon = Epsilon
        self.C=[]
        self.U=[]


    def import_data(file, delimiter):
    	"""
    	 This function imports the data into a list form a file name passed as an argument.
    	"""
    	data = []
    	cluster_location =[]
    	f = open(str(file), 'r')
    	count=0
    	for line in f:
    		if(count==0):
    			count+=1
    			continue;

    		current = line.split(delimiter)
    		current_dummy = []
    		for j in range(1,len(current)):
    			current_dummy.append(float(current[j]))
    		j+=1
    		cluster_location.append(current[0])
    		data.append(current_dummy)

    	print ("finished importing data")
    	return data

    def randomise_data(data):
    	"""
    	This function randomises the data, and also keeps record of the order of randomisation.
    	"""
    	order = list(range(0,len(data)))
    	random.shuffle(order)
    	new_data = [[] for i in range(0,len(data))]
    	for index in range(0,len(order)):
    		new_data[index] = data[order[index]]
    	return new_data, order

    def de_randomise_data(data, order):
    	"""
    	This function would return the original order of the data, pass the order list returned in randomise_data() as an argument
    	"""
    	new_data = [[]for i in range(0,len(data))]
    	for index in range(len(order)):
    		new_data[order[index]] = data[index]
    	return new_data

    def print_matrix(list):
    	"""
    	Prints the matrix in a more reqdable way
    	"""
    	for i in range(0,len(list)):
    		print (list[i])

    def end_conditon(self, U,U_old):
    	"""
    	This is the end conditions, it happens when the U matrix stops chaning too much with successive iterations.
    	"""
    	for i in range(0,len(U)):
    		for j in range(0,len(U[0])):
    			if abs(U[i][j] - U_old[i][j]) > self.Epsilon :
    				return False
    	return True

    def initialise_U(self, data, cluster_number):
    	"""
    	This function would randomis U such that the rows add up to 1. it requires a global MAX.
    	"""
    	global MAX
    	U = []
    	for i in range(0,len(data)):
    		current = []
    		rand_sum = 0.0
    		for j in range(0,cluster_number):
    			dummy = random.randint(1,int(self.MAX))
    			current.append(dummy)
    			rand_sum += dummy
    		for j in range(0,cluster_number):
    			current[j] = current[j] / rand_sum
    		U.append(current)
    	return U

    def distance(self, point, center):
    	"""
    	This function calculates the distance between 2 points (taken as a list). We are refering to Eucledian Distance.
    	"""
    	if len(point) != len(center):
    		return -1
    	dummy = 0.0
    	for i in range(0,len(point)):
    		dummy += abs(point[i] - center[i]) ** 2
    	return math.sqrt(dummy)

    def normalise_U(self, U):
    	"""
    	This de-fuzzifies the U, at the end of the clustering. It would assume that the point is a member of the cluster whoes membership is maximum.
    	"""
    	for i in range(0,len(U)):
    		maximum = max(U[i])
    		for j in range(0,len(U[0])):
    			if U[i][j] != maximum:
    				U[i][j] = 0
    			else:
    				U[i][j] = 1
    	return U


    def fuzzy_train(self, data, cluster_number, m = 2):
    	"""
    	This function would calculate the required center, return the final normalised membership matrix U and the centers C.
    	It's paramaters are the : cluster number and the fuzzifier "m".
    	Used for training
    	"""
    	## initialise the U matrix:
    	U = self.initialise_U(data, cluster_number)

    	#initilise the loop
    	while (True):
    		#create a copy of it, to check the end conditions
    		U_old = copy.deepcopy(U)
    		# cluster center vector
    		C = []
    		for j in range(0,cluster_number):
    			current_cluster_center = []
    			for i in range(0,len(data[0])): #this is the number of dimensions
    				dummy_sum_num = 0.0
    				dummy_sum_dum = 0.0
    				for k in range(0,len(data)):
    					dummy_sum_num += (U[k][j] ** m) * data[k][i]
    					dummy_sum_dum += (U[k][j] ** m)
    				current_cluster_center.append(dummy_sum_num/dummy_sum_dum)
    			C.append(current_cluster_center)

    		#creating a distance vector, useful in calculating the U matrix.
    		distance_matrix =[]
    		for i in range(0,len(data)):
    			current = []
    			for j in range(0,cluster_number):
    				current.append(self.distance(data[i], C[j]))
    			distance_matrix.append(current)

    		# update U vector
    		for j in range(0, cluster_number):
    			for i in range(0, len(data)):
    				dummy = 0.0
    				for k in range(0,cluster_number):
    					dummy += (distance_matrix[i][j]/ distance_matrix[i][k]) ** (2/(m-1))
    				U[i][j] = 1 / dummy

    		if self.end_conditon(U,U_old):
    			break

    	U = self.normalise_U(U)

    	return U, C

    def fuzzy_predict(self, data, cluster_number, C, m):
    	"""
    	This function would return the final normalised membership matrix U about the test set.
    	It's paramaters are the : cluster number, previous centroids and the fuzzifier "m".
    	"""

    	self.U = self.initialise_U(data, cluster_number)
    	#creating a distance vector, useful in calculating the U matrix.
    	distance_matrix =[]
    	for i in range(0,len(data)):
    		current = []
    		for j in range(0,cluster_number):
    			current.append(self.distance(data[i], C[j]))
    		distance_matrix.append(current)

    	# update U vector
    	for j in range(0, cluster_number):
    		for i in range(0, len(data)):
    			dummy = 0.0
    			for k in range(0,cluster_number):
    				dummy += (distance_matrix[i][j]/ distance_matrix[i][k]) ** (2/(m-1))
    			self.U[i][j] = 1 / dummy
    	return self.U

    def RMSE_membership(self, test_membership, y_test):
        #RMSE calcolato sulla predizione della membership
        res=[]
        res2=[]
        err = []

        for i in range(0, len(test_membership)):
            res.append(test_membership[i][0])
            res2.append(test_membership[i][1])

        mse = mean_squared_error(y_test, res, squared=False)
        mse2 = mean_squared_error(y_test, res2, squared=False)
        err.append(min(mse, mse2))
        return np.mean(err)

    def accuracy(self, test_membership, y_test):
        new_U = self.normalise_U(self.U)
        res = []
        for i in range(0, len(new_U)):
        	res.append(new_U[i][0])

        res=[]
        res2=[]
        acc=0
        acc2=0

        for i in range(0, len(test_membership)):
            imax=test_membership[i].index(max(test_membership[i]))
            res.append(imax)
            res2.append((imax+1)%2)

        for i in range(0, len(test_membership)):
            if(res[i]==y_test[i]):
                acc+=1

        for i in range(0, len(test_membership)):
            if(res2[i]==y_test[i]):
                acc2+=1

        return max(acc, acc2)/len(y_test)


    def RMSE(self, test_membership, y_test):
        res=[]
        res2=[]
        res3=[]
        mse=0
        mse2=0
        mse3=0

        for i in range(0, len(test_membership)):
            imax=test_membership[i].index(max(test_membership[i]))
            res.append(imax)
            res2.append((imax+1)%3)
            res3.append((imax+2)%3)

        mse = mean_squared_error(y_test, res, squared=False)
        mse2 = mean_squared_error(y_test, res2, squared=False)
        mse3 = mean_squared_error(y_test, res3, squared=False)

        return min(mse, mse2, mse3)
