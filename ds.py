import numpy as np
from sklearn import svm, metrics

def load_data(filename):
    data = np.genfromtxt (filename, delimiter=",")
    data =  np.delete(data,0,0) #delete header row
    data =  np.delete(data,0,1) #delete id column
    labels = data[:,256] #labels
    data =  np.delete(data,256,axis=1) #delete labels column
    return data,labels


tdata, tlabels = load_data('Train.csv')
vdata, vlabels = load_data('Validate.csv')

print(vdata.shape)
print(sum(vlabels))
print(tdata.shape)
print(sum(tlabels))

np.random.shuffle(tdata)
print("shuffled")

classifier = svm.SVC(gamma=0.001,C=1.0, class_weight='balanced',kernel='rbf') #higher C => Overfitting
classifier.fit(tdata, tlabels)

expected = vlabels
predicted = classifier.predict(vdata)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

