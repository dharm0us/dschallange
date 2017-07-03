import numpy as np
from sklearn import svm, metrics

csv = np.genfromtxt ('Train.csv', delimiter=",")
csv =  np.delete(csv,0,0) #delete header row
csv =  np.delete(csv,0,1) #delete id column
labels = csv[:,256] #labels
csv =  np.delete(csv,256,axis=1) #delete labels column
print(csv.shape)
print(csv)
print(labels)
print(np.unique(labels))

n_samples = len(csv)
classifier = svm.SVC(gamma=0.001)
classifier.fit(csv[:n_samples // 2], labels[:n_samples // 2])
expected = labels[n_samples // 2:]
predicted = classifier.predict(csv[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

