import numpy as np
import time
from time import gmtime, strftime
from sklearn import svm, metrics
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

def load_data(filename):
    data = np.genfromtxt (filename, delimiter=",")
    data =  np.delete(data,0,0) #delete header row
    data =  np.delete(data,0,1) #delete id column
    labels = data[:,256] #labels
    data =  np.delete(data,256,axis=1) #delete labels column
    return data,labels

def gt():
    return strftime("%Y-%m-%d %H:%M:%S ", gmtime())

print(gt()+"data load start")
tdata, tlabels = load_data('Train.csv')
scaler = preprocessing.StandardScaler().fit(tdata)
scaler.transform(tdata)

vdata, vlabels = load_data('Validate.csv')
scaler.transform(vdata)

print(gt()+"data load end")

print(tdata.shape)
print(sum(tlabels))
print(vdata.shape)
print(sum(vlabels))

#np.random.shuffle(tdata)
#print("shuffled")

print(gt()+"fit start")
classifier = svm.SVC(gamma=0.001,C=1.0, kernel='rbf',class_weight='balanced') #higher C => Overfitting
classifier.fit(tdata, tlabels)
print(gt()+"fit complete")

expected = vlabels
predicted = classifier.predict(vdata)

print(gt()+"Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print(gt()+"Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


# With PP, with class_balanced, proper scaling
# 2017-07-03 14:47:41 Classification report for classifier SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
#   decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=0.001, verbose=False):
#              precision    recall  f1-score   support
#
#         0.0       0.94      0.80      0.87      1881
#         1.0       0.87      0.97      0.92      2686
#
# avg / total       0.90      0.90      0.90      4567
#
#
# 2017-07-03 14:47:41 Confusion matrix:
# [[1507  374]
#  [  90 2596]]

# With PP, with class_balanced, simnple PP -> preprocessing.Scale()
# Classification report for classifier SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
#   decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=0.001, verbose=False):
#              precision    recall  f1-score   support
#
#         0.0       0.96      0.83      0.89      1881
#         1.0       0.89      0.97      0.93      2686
#
# avg / total       0.92      0.91      0.91      4567
#
#
# Confusion matrix:
# [[1559  322]
#  [  70 2616]]
#accuracy = 0.9141

#With PP, no class_balance
# 2017-07-03 13:51:20 Classification report for classifier SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#   decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=0.001, verbose=False):
#              precision    recall  f1-score   support
#
#         0.0       0.97      0.76      0.85      1881
#         1.0       0.85      0.99      0.91      2686
#
# avg / total       0.90      0.89      0.89      4567
#
#
# 2017-07-03 13:51:20 Confusion matrix:
# [[1426  455]
#  [  38 2648]]

# No PP, no class_balance
# 2017-07-03 13:47:54 Classification report for classifier SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#   decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=0.001, verbose=False):
#              precision    recall  f1-score   support
#
#         0.0       0.97      0.66      0.79      1881
#         1.0       0.81      0.99      0.89      2686
#
# avg / total       0.88      0.85      0.85      4567
#
#
# 2017-07-03 13:47:54 Confusion matrix:
# [[1245  636]
#  [  33 2653]]

 #No PP, with class_balanced
# 2017-07-03 13:44:23 Classification report for classifier SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
#   decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=0.001, verbose=False):
#              precision    recall  f1-score   support
#
#         0.0       0.94      0.80      0.87      1881
#         1.0       0.87      0.97      0.92      2686
#
# avg / total       0.90      0.90      0.90      4567
#
#
# 2017-07-03 13:44:23 Confusion matrix:
# [[1507  374]
#  [  90 2596]]