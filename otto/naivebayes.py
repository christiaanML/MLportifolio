#author: ChristiaanML

import numpy, os, pdb

def loadData(file_in):

	data = numpy.loadtxt(file_in,dtype = str, skiprows = 1, delimiter = ',')
	ID = data[:,0].astype(numpy.int)
	data_features = data[:,1:-1].astype(numpy.float)
	data_labels = numpy.char.replace(data[:,-1],'Class_','').astype(numpy.int)
	
	return 	data_features,ID, data_labels
	
data_features,ID, data_labels = loadData(os.path.join("data","train.csv"))

idx = numpy.argsort(numpy.random.rand(data_features.shape[0])) # random idx.

#cross validation 
Ncross=5
for j in range(Ncross):

	Nstart = int(j*data_features.shape[0]/Ncross)
	Nend = min(int((j+1)*data_features.shape[0]/Ncross),data_features.shape[0])
	bool_test = numpy.logical_and(numpy.arange(data_features.shape[0])>=Nstart,numpy.arange(data_features.shape[0])<Nend)
	bool_train= numpy.logical_not(bool_test)
	# split in train/test data.
	train_data = data_features[idx[bool_train],:]
	train_ID = ID[idx[bool_train]]
	train_labels = data_labels[idx[bool_train]]
	test_data = data_features[idx[bool_test],:]
	test_ID = ID[idx[bool_test]]
	test_labels = data_labels[idx[bool_test]]


	unique_labels = numpy.unique(data_labels)

	#determine mu and sigmas of Gauss Distributions and determine probabilities of classes.
	mu_array = numpy.zeros((len(unique_labels),data_features.shape[1]), numpy.float)
	sigma_array = numpy.zeros((len(unique_labels),data_features.shape[1]), numpy.float)
	PC = numpy.zeros(len(unique_labels),numpy.float)
	for i in range(len(unique_labels)):
		bool=train_labels==unique_labels[i]
		mu_array[i,:]=train_data[bool,:].mean(axis=0)
		sigma_array[i,:]=numpy.std(train_data[bool,:],axis=0)
		PC[i]=float(sum(bool))/train_data.shape[0]


	# calculate maximum likelihood
	bayes = numpy.zeros((test_data.shape[0], len(unique_labels)),numpy.float)
	for i in range(len(unique_labels)):
		
		I1 = sigma_array[i,:]!=0# filter for std zero.
		Gauss = 1/(numpy.sqrt(2*numpy.pi)*sigma_array[i,I1])*numpy.exp(-0.5*(test_data[:,I1]-mu_array[i,I1])**2/sigma_array[i,I1]**2)
		bayes[:,i]=numpy.prod(Gauss,axis=1)*PC[i]

	labels_pred = numpy.argmax(bayes,axis=1)+1
	print(float(sum(abs(labels_pred==test_labels)))/test_labels.shape[0])

