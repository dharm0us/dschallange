import numpy as np
from sklearn import svm, metrics

csv = np.genfromtxt ('Train.csv', delimiter=",")
csv =  np.delete(csv,0,0) #delete header row
csv =  np.delete(csv,0,1) #delete id column
#np.random.shuffle(csv)
labels = csv[:,256] #labels
csv =  np.delete(csv,256,axis=1) #delete labels column
print(csv.shape)
print(csv)
print(labels)
print(sum(labels))
print(np.unique(labels))

n_samples = len(csv)
classifier = svm.SVC(gamma=0.001,class_weight='balanced')
classifier.fit(csv[:n_samples // 2], labels[:n_samples // 2])
expected = labels[n_samples // 2:]
predicted = classifier.predict(csv[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

# commit id 2604ec9
# Classification report for classifier SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#   decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=0.001, verbose=False):
#              precision    recall  f1-score   support
#
#         0.0       0.96      0.29      0.45      2336
#         1.0       0.85      1.00      0.92      9410
#
# avg / total       0.87      0.86      0.82     11746
#
#
# Confusion matrix:
# [[ 682 1654]
#  [  30 9380]]

# fe629c6
# Classification report for classifier SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
#   decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=0.001, verbose=False):
#              precision    recall  f1-score   support
#
#         0.0       0.84      0.80      0.82      2336
#         1.0       0.95      0.96      0.96      9410
#
# avg / total       0.93      0.93      0.93     11746
#
#
# Confusion matrix:
# [[1877  459]
#  [ 349 9061]]