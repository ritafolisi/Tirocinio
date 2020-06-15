from FCM import *


model = FCM()
dataset=str(sys.argv[1])
accuracy = model.train(dataset);
acc=model.test(dataset)
