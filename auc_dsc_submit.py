import numpy as np
import time
from time import gmtime, strftime

import sys
from sklearn import svm, metrics
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

# g = 0.0002
# c = 1.0
# b = 'balanced'
# 0.041416041514
# 0.958583958486

def load_test_data(filename):
    data = np.genfromtxt (filename, delimiter=",")
    data =  np.delete(data,0,0) #delete header row
    data =  np.delete(data,0,1) #delete id column
    return data

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
    f = open('res_auc_submit.txt', 'a')
    f.write(msg+"\n")  # python will convert \n to os.linesep
    f.close()  # you can omit in most cases as the destructor will call it

def run(tdata,tlabels,vdata):
    g = 0.0002
    c = 1.0
    b = 0.0 
    flog(gt()+"fit start")
    if(b<1):
        cw = 'balanced'
    else:
        cw = {0: b}
    tlabels = tlabels.flatten()
    print(tlabels.shape)

    n_classes = 2

    random_state = np.random.RandomState(0)

    flog("creating classifier")
    classifier = svm.SVC(kernel='rbf', probability=True, random_state=random_state,gamma=g,C=c,class_weight=cw)
    flog("scoring")
    y_score = classifier.fit(tdata, tlabels).predict_proba(vdata)
    print(y_score)
    np.savetxt("foo.csv", y_score, delimiter=",")


    # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(vlabels.ravel(), y_score.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # classifier = svm.SVC(gamma=g,C=c, kernel='rbf',class_weight=cw) #higher C => Overfitting
    # classifier.fit(tdata, tlabels)
    # flog(gt()+"fit complete")
    #
    # expected = vlabels
    # predicted = classifier.predict(vdata)
    #
    # accuracy = (cm[0,0]+cm[1,1])*100.0/sum(sum(cm))
    # flog(accuracy)
    # flog(gt()+"Confusion matrix:\n%s" % cm)
    # sys.stdout.flush()
    # return accuracy

flog(gt()+"data load start")
tdata, tlabels = load_data('Train.csv')
scaler = preprocessing.StandardScaler().fit(tdata)
tdata = scaler.transform(tdata)

vdata = load_test_data('TestData.csv')
vdata = scaler.transform(vdata)
print(tdata.shape)
print(vdata.shape)

flog(gt()+"data load end")

flog(tdata.shape)
flog(sum(tlabels))
flog(vdata.shape)

curr = run(tdata,tlabels,vdata)
