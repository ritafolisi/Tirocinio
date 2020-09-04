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

#used for randomising U
global MAX
MAX = 10000.0
#used for end condition
global Epsilon
Epsilon = 0.00000001

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

def end_conditon(U,U_old):
	"""
	This is the end conditions, it happens when the U matrix stops chaning too much with successive iterations.
	"""
	global Epsilon
	for i in range(0,len(U)):
		for j in range(0,len(U[0])):
			if abs(U[i][j] - U_old[i][j]) > Epsilon :
				return False
	return True

def initialise_U(data, cluster_number):
	"""
	This function would randomis U such that the rows add up to 1. it requires a global MAX.
	"""
	global MAX
	U = []
	for i in range(0,len(data)):
		current = []
		rand_sum = 0.0
		for j in range(0,cluster_number):
			dummy = random.randint(1,int(MAX))
			current.append(dummy)
			rand_sum += dummy
		for j in range(0,cluster_number):
			current[j] = current[j] / rand_sum
		U.append(current)
	return U

def distance(point, center):
	"""
	This function calculates the distance between 2 points (taken as a list). We are refering to Eucledian Distance.
	"""
	if len(point) != len(center):
		return -1
	dummy = 0.0
	for i in range(0,len(point)):
		dummy += abs(point[i] - center[i]) ** 2
	return math.sqrt(dummy)

def normalise_U(U):
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


def fuzzy_train(data, cluster_number, m = 2):
	"""
	This function would calculate the required center, return the final normalised membership matrix U and the centers C.
	It's paramaters are the : cluster number and the fuzzifier "m".
	Used for training
	"""
	## initialise the U matrix:
	U = initialise_U(data, cluster_number)

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
				current.append(distance(data[i], C[j]))
			distance_matrix.append(current)

		# update U vector
		for j in range(0, cluster_number):
			for i in range(0, len(data)):
				dummy = 0.0
				for k in range(0,cluster_number):
					dummy += (distance_matrix[i][j]/ distance_matrix[i][k]) ** (2/(m-1))
				U[i][j] = 1 / dummy

		if end_conditon(U,U_old):
			break

	U = normalise_U(U)

	return U, C

def fuzzy_predict(data, cluster_number, C, m):
	"""
	This function would return the final normalised membership matrix U about the test set.
	It's paramaters are the : cluster number, previous centroids and the fuzzifier "m".
	"""

	U = initialise_U(data, cluster_number)

	#creating a distance vector, useful in calculating the U matrix.
	distance_matrix =[]
	for i in range(0,len(data)):
		current = []
		for j in range(0,cluster_number):
			current.append(distance(data[i], C[j]))
		distance_matrix.append(current)

	# update U vector
	for j in range(0, cluster_number):
		for i in range(0, len(data)):
			dummy = 0.0
			for k in range(0,cluster_number):
				dummy += (distance_matrix[i][j]/ distance_matrix[i][k]) ** (2/(m-1))
			U[i][j] = 1 / dummy

	#U = normalise_U(U)

	return U

## main
if __name__ == '__main__':

	# import the data
	data=sys.argv[1]
	#Da utilizzare se so a pripri i nomi delle colonne
	#feat1=sys.argv[2]
	#feat2=sys.argv[3]
	#labels=sys.argv[4]

	dataset=pd.read_csv(data)

	# extract features and labels
	#X = dataset[[feat1, feat2]].values
	#y = dataset[labels].values

	X = dataset.iloc[:, 1:3]
	y = dataset.iloc[:, 0]
	X = np.asarray(X)
	y = np.asarray(y)
	#print(X, y)

	N_SPLITS = 5;
	err = []

	#cross validation
	skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=1)
	for train_index, test_index in skf.split(X, y):
		#print("TRAIN:", train_index, "\nTEST:", test_index)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		#training
		train_membership, centers = fuzzy_train(X_train , 2 , 2)

		#test
		test_membership = fuzzy_predict(X_test , 2 , centers, 2)

		#MSE calcolato sulla predizione della membership
		res=[]
		res2=[]
		for i in range(0, len(test_membership)):
			res.append(test_membership[i][0])
			res2.append(test_membership[i][1])


		#MSE, calcolato sulle label predette (scommentare normalize_U in fuzzy_predict)
		#res=[]
		#res2=[]
		#for i in range(0, len(test_membership)):
		#	imax=test_membership[i].index(max(test_membership[i]))
		#	res.append(imax)
		#	res2.append((imax+1)%2)

		mse = mean_squared_error(y_test, res, squared=False)
		mse2 = mean_squared_error(y_test, res2, squared=False)
		err.append(min(mse, mse2))

	print("Errore:", np.mean(err))
