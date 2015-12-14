import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn import svm, grid_search

print "Preparing the data..."
print "--------------------------------"
print "Loading the train dataset..."
train = np.loadtxt("train.csv", delimiter=",", skiprows=1)
print "Loading the test dataset..."
test_data = np.loadtxt("test.csv", delimiter=",", skiprows=1)

print "separate labels from training data"
train_data = train[:,1:]
target = train[:,0]

train_data_normalized = preprocessing.normalize(train_data, norm='l2')
test_data_normalized = preprocessing.normalize(test_data, norm='l2')
#number of components to extract
print "Reduction ..."
pca = PCA(n_components=40, whiten=True)
pca.fit(X_normalized)
print "transform training data..."
train_data_normalized = pca.transform(train_data_normalized)
print "transform test data..."
test_data_normalized = pca.transform(test_data_normalized)

print "Choose best hyperparameters..."
gammas = np.logspace(-2,3,15)
cs = np.logspace(-2,3,15)

svc = svm.SVC(kernel='rbf')
clf = grid_search.GridSearchCV(estimator=svc, param_grid=[dict(gamma=gammas), dict(C=cs)],n_jobs=-1)
clf.fit(train_data, target)

print "best hyperparameters:\nC:{0}\ngamma:{1}".format(clf.best_estimator_.C, clf.best_estimator_.gamma)

print "apply svm with best hyperparameters"
svc2 = svm.SVC(gamma=clf.best_estimator_.gamma, C=clf.best_estimator_.C)
svc2.fit(train_data_normalized, target)
test_y = svc2.predict(test_data_normalized)
pd.DataFrame({"ImageId": range(1,len(test_y)+1), "Label": map(int,test_y)}).to_csv('hyper_parameters_svm_normalized.csv', index=False, header=True)
