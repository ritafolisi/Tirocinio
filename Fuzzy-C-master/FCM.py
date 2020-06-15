###############################################################################
##
##	Ananya Kirti @ June 9 2015
##	Fuzzy C means
##
###############################################################################
## Ananya Kirti
# importing all the required components, you may also use scikit for a direct implementation.
import copy
import math
import random
import time
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import decimal

class FCM():
    def __init__(self, MAX=10000.0, Epsilon=0.00000001):
        self.MAX = MAX
        self.Epsilon = Epsilon
        self.C=[]
        self.U=[]

    def import_data_format_iris(self,file):
    	"""
    	This would format the data as required by iris
    	the link for the same is http://archive.ics.uci.edu/ml/machine-learning-databases/iris/
    	"""
    	data = []
    	cluster_location =[]
    	f = open(str(file), 'r')
    	for line in f:
    		if(line.startswith("sepal_length")):
    			continue
    		else:
    			current = line.split(",")
    			current_dummy = []
    			for j in range(0,len(current)-1):
    				current_dummy.append(float(current[j]))
    			j+=1
    			#print current[j]
    			if  (current[j] == "Iris-setosa\n" or current[j] == "setosa\n"):
    				cluster_location.append(0)
    			elif (current[j] == "Iris-versicolor\n" or current[j] == "versicolor\n"):
    				cluster_location.append(1)
    			else:
    				cluster_location.append(2)
    			data.append(current_dummy)
    	print ("finished importing data")
    	return data , cluster_location

    def randomise_data(self, data):
    	"""
    	This function randomises the data, and also keeps record of the order of randomisation.
    	"""
    	order = list(range(0,len(data)))
    	random.shuffle(order)
    	new_data = [[] for i in range(0,len(data))]
    	for index in range(0,len(order)):
    		new_data[index] = data[order[index]]
    	return new_data, order

    def de_randomise_data(self, data, order):
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

    def end_conditon(self,U,U_old):
    	"""
    	This is the end conditions, it happens when the U matrix stops chaning too much with successive iterations.
    	"""
    	global Epsilon
    	for i in range(0,len(U)):
    		for j in range(0,len(U[0])):
    			if abs(U[i][j] - U_old[i][j]) > self.Epsilon :
    				return False
    	return True

    def initialise_U(self,data, cluster_number):
    	"""
    	This function would randomis U such that the rows add up to 1. it requires a global MAX.
    	"""
    	global MAX
    	for i in range(0,len(data)):
    		current = []
    		rand_sum = 0.0
    		for j in range(0,cluster_number):
    			dummy = random.randint(1,int(self.MAX))
    			current.append(dummy)
    			rand_sum += dummy
    		for j in range(0,cluster_number):
    			current[j] = current[j] / rand_sum
    		self.U.append(current)
    	return self.U

    def distance(self,point, center):
    	"""
    	This function calculates the distance between 2 points (taken as a list). We are refering to Eucledian Distance.
    	"""
    	if len(point) != len(center):
    		return -1
    	dummy = 0.0
    	for i in range(0,len(point)):
    		dummy += abs(point[i] - center[i]) ** 2
    	return math.sqrt(dummy)

    def normalise_U(self):
    	"""
    	This de-fuzzifies the U, at the end of the clustering. It would assume that the point is a member of the cluster whoes membership is maximum.
    	"""
    	for i in range(0,len(self.U)):
    		maximum = max(self.U[i])
    		for j in range(0,len(self.U[0])):
    			if self.U[i][j] != maximum:
    				self.U[i][j] = 0
    			else:
    				self.U[i][j] = 1
    	return self.U

    def checker_iris(self,final_location):
    	"""
    	This is used to find the percentage correct match with the real clustering.
    	"""
    	right = 0.0
    	for k in range(0,3):
    		checker =[0,0,0]
    		for i in range(0,50):
    			for j in range(0,len(final_location[0])):
    				if final_location[i + (50*k)][j] == 1:
    					checker[j] += 1
    		right += max(checker)
    		print (right)
    	answer =  right / 150 * 100
    	return str(answer) +  " % accuracy"

    def fuzzy(self,data, cluster_number, m = 2):
    	"""
    	This is the main function, it would calculate the required center, and return the final normalised membership matrix U.
    	It's paramaters are the : cluster number and the fuzzifier "m".
    	"""
    	## initialise the U matrix:
    	self.U = self.initialise_U(data, cluster_number)
    	#print_matrix(U)
    	#initilise the loop
    	while (True):
    		#create a copy of it, to check the end conditions
    		U_old = copy.deepcopy(self.U)
    		# cluster center vector
    		self.C = []
    		for j in range(0,cluster_number):
    			current_cluster_center = []
    			for i in range(0,len(data[0])): #this is the number of dimensions
    				dummy_sum_num = 0.0
    				dummy_sum_dum = 0.0
    				for k in range(0,len(data)):
    					dummy_sum_num += (self.U[k][j] ** m) * data[k][i]
    					dummy_sum_dum += (self.U[k][j] ** m)
    				current_cluster_center.append(dummy_sum_num/dummy_sum_dum)
    			self.C.append(current_cluster_center)


    		#creating a distance vector, useful in calculating the U matrix.

    		distance_matrix =[]
    		for i in range(0,len(data)):
    			current = []
    			for j in range(0,cluster_number):
    				current.append(self.distance(data[i], self.C[j]))
    			distance_matrix.append(current)

    		# update U vector
    		for j in range(0, cluster_number):
    			for i in range(0, len(data)):
    				dummy = 0.0
    				for k in range(0,cluster_number):
    					dummy += (distance_matrix[i][j]/ distance_matrix[i][k]) ** (2/(m-1))
    				self.U[i][j] = 1 / dummy

    		if self.end_conditon(self.U,U_old):
    			print ("finished clustering")
    			break
    	U = self.normalise_U()
    	print ("normalised U")
    	return U

    def train(self, dataset):
    	data, cluster_location = self.import_data_format_iris(dataset);
    	data , order = self.randomise_data(data)

    	start = time.time()
    	final_location = self.de_randomise_data(self.fuzzy(data, 2, 2), order)
        #final_location = self.de_randomise_data(final_location, order)

    	accuracy = self.checker_iris(final_location)
    	accuracy = accuracy.split(" ")[0]
    	accuracy = float(accuracy)
    	return self, accuracy

    def test(self, dataset):
        data, cluster_location = self.import_data_format_iris(dataset);
        data, order = self.randomise_data(data)
        final_location=self.de_randomise_data(self.U, order)
        m=2
        cluster_number=2
    	#prendi data e lavoraci
        distance_matrix =[]
        for i in range(0,len(data)):
            current = []
            for j in range(0,cluster_number):
                current.append(self.distance(data[i], self.C[j]))
            distance_matrix.append(current)

    	# update U vector
        for j in range(0, cluster_number):
            for i in range(0, len(data)):
                dummy = 0.0
                for k in range(0,cluster_number):
                    dummy += (distance_matrix[i][j]/ distance_matrix[i][k]) ** (2/(m-1))
                self.U[i][j] = 1 / dummy

        self.U = self.normalise_U()

        accuracy = self.checker_iris(final_location)
        accuracy = accuracy.split(" ")[0]
        accuracy = float(accuracy)
        return accuracy
