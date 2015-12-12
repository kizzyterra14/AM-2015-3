import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC

print "Preparing the data..."
print "--------------------------------"
print "Loading the train dataset..."
train = np.loadtxt("train.csv", delimiter=",", skiprows=1)
print "Loading the test dataset..."
test_data = np.loadtxt("test.csv", delimiter=",", skiprows=1)

print "separate labels from training data"
train_data = train[:,1:]
target = train[:,0]

#number of components to extract
print "Reduction ..."
pca = PCA(n_components=35, whiten=True)
pca.fit(train_data)
train_data = pca.transform(train_data)


print "Training SVM..."
svc = SVC(verbose=True)
svc.fit(train_data, target)

test_data = pca.transform(test_data)
test_y = svc.predict(test_data)


print "Saving predictions..."
# with open('svm_pca.csv', 'w') as writer:
#     writer.write('"ImageId","Label"\n')
#     count = 0
#     for p in predict:
#         count += 1
#         writer.write(str(count) + ',"' + str(p) + '"\n')
pd.DataFrame({"ImageId": range(1,len(test_y)+1), "Label": map(int,test_y)}).to_csv('svm_l.csv', index=False, header=True)

# O melhor resultado foi obtido com PCA 35 componentes, C=1
