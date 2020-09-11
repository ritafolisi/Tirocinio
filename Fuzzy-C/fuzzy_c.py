from FCM import *
import logging
import numpy as np

def fcm_script(data):

	#Da utilizzare se so a priori i nomi delle colonne
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

	model = FCM()

	N_SPLITS = 5;
	error=[]
	score=[]

	#cross validation
	skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=1)
	for train_index, test_index in skf.split(X, y):
		#print("TRAIN:", train_index, "\nTEST:", test_index)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		#training
		train_membership, centers = model.fuzzy_train(X_train , 2 , 2)

		#test
		test_membership = model.fuzzy_predict(X_test , 2 , centers, 2)

		error.append(model.RMSE_membership(test_membership, y_test))
		score.append(model.accuracy(test_membership, y_test))
		#score=77
	return model, score, error


def main():

	logging.basicConfig(filename = 'esperimenti.log', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
	# import data
	data=sys.argv[1]

	logging.info('Questo esperimento lavora sul dataset: %s', data)

	model, score, error = fcm_script(data)
	logging.info(f'{score} = array delle accuratezze')
	logging.info(f'{error} = array delle RMSE_membership \n')
	print(score)
	print (error)

## main
if __name__ == '__main__':
	main()
