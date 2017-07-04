import numpy as np
import time
from time import gmtime, strftime

import sys
from sklearn import svm, metrics
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize


def load_data(filename):
    data = np.genfromtxt (filename, delimiter=",")
    data =  np.delete(data,0,0) #delete header row
    data =  np.delete(data,0,1) #delete id column
    labels = data[:,256] #labels
    data =  np.delete(data,256,axis=1) #delete labels column
    return data,labels

def gt():
    return strftime("%Y-%m-%d %H:%M:%S ", gmtime())

def flog(msg):
    msg = str(msg)
    print(msg)
    f = open('res.txt', 'a')
    f.write(msg+"\n")  # python will convert \n to os.linesep
    f.close()  # you can omit in most cases as the destructor will call it

def run(tdata,tlabels,vdata,vlabels,g,c,b):
    flog(gt()+"fit start")
    if(b<1):
        cw = 'balanced'
    else:
        cw = {0: b}
    tlabels = tlabels.flatten()
    tlabels = label_binarize(tlabels, classes=[0, 1])

    vlabels = vlabels.flatten()
    vlabels = label_binarize(vlabels, classes=[0, 1])

    n_classes = tlabels.shape[1]

    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = tdata.shape
    tdata = np.c_[tdata, random_state.randn(n_samples, 200 * n_features)]
    vn_samples, vn_features = vdata.shape
    vdata = np.c_[vdata, random_state.randn(vn_samples, 200 * vn_features)]

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True, random_state=random_state))
    y_score = classifier.fit(tdata, tlabels).decision_function(vdata)
    print y_score

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
     fpr[i], tpr[i], _ = roc_curve(vlabels[:, i], y_score[:, i])
     roc_auc[i] = auc(fpr[i], tpr[i])
    print roc_auc

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(vlabels.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # classifier = svm.SVC(gamma=g,C=c, kernel='rbf',class_weight=cw) #higher C => Overfitting
    # classifier.fit(tdata, tlabels)
    # flog(gt()+"fit complete")
    #
    # expected = vlabels
    # predicted = classifier.predict(vdata)
    #
    # flog(gt()+"Classification report for classifier %s:\n%s\n"
    #      % (classifier, metrics.classification_report(expected, predicted)))
    # cm = metrics.confusion_matrix(expected, predicted)
    # accuracy = (cm[0,0]+cm[1,1])*100.0/sum(sum(cm))
    # flog(accuracy)
    # flog(gt()+"Confusion matrix:\n%s" % cm)
    # sys.stdout.flush()
    # return accuracy

flog(gt()+"data load start")
tdata, tlabels = load_data('Validate.csv')
scaler = preprocessing.StandardScaler().fit(tdata)
tdata = scaler.transform(tdata)

vdata, vlabels = load_data('TrainSample.csv')
vdata = scaler.transform(vdata)

flog(gt()+"data load end")

flog(tdata.shape)
flog(sum(tlabels))
flog(vdata.shape)
flog(sum(vlabels))

accuracy = 0
best_g = 0
best_c = 0
best_b = 0
for g in np.arange(0.0001, 0.0005, 0.0001):
    for c in np.arange(1.0, 5.0, 1.0):
        for b in np.arange(0.0, 4.0, 1.0):
            curr = run(tdata,tlabels,vdata,vlabels,g,c,b)
            if(curr > accuracy):
                accuracy = curr
                best_g = g
                best_c = c
                best_b = b
                flog("best so far "+str(accuracy)+" for "+str(g)+" and "+str(c)+" and "+str(b))
                exit(1)
            flog("best so far "+str(accuracy)+" for "+str(best_g)+" and "+str(best_c)+" and "+str(best_b))

        #np.random.shuffle(tdata)
        #print("shuffled")


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
