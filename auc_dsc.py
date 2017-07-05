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
    f = open('res_auc.txt', 'a')
    f.write(msg+"\n")  # python will convert \n to os.linesep
    f.close()  # you can omit in most cases as the destructor will call it

def run(tdata,tlabels,vdata,vlabels,g,c,b):
    # g = 0.0002
    # c = 1.0
    # b = 'balanced'
    flog(gt()+"fit start")
    if(b<1):
        cw = 'balanced'
    else:
        cw = {0: b}
    tlabels = tlabels.flatten()
    #tlabels = label_binarize(tlabels, classes=[0, 1])
    print tlabels.shape
    vlabels = vlabels.flatten()
    #labels = label_binarize(vlabels, classes=[0, 1])
    print vlabels.shape
    #exit(1)

    n_classes = 2

    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    # n_samples, n_features = tdata.shape
    # tdata = np.c_[tdata, random_state.randn(n_samples, 200 * n_features)]
    # vn_samples, vn_features = vdata.shape
    # vdata = np.c_[vdata, random_state.randn(vn_samples, 200 * vn_features)]

    # Learn to predict each class against the other
    flog("creating classifier")
    #classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True, random_state=random_state))
    classifier = svm.SVC(kernel='rbf', probability=True, random_state=random_state,gamma=g,C=c,class_weight=cw)
    flog("scoring")
    y_score = classifier.fit(tdata, tlabels).predict_proba(vdata)
    print y_score

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    print vlabels.shape
    print y_score.shape
    print y_score[:,0].shape
    fpr, tpr, _ = roc_curve(vlabels, y_score[:,0])
    roc_auc_1 = auc(fpr, tpr)
    flog(roc_auc_1)
    fpr, tpr, _ = roc_curve(vlabels, y_score[:,1])
    roc_auc_2 = auc(fpr, tpr)
    flog(roc_auc_2)
    flog(gt()+"Classification report for classifier %s:\n" % (classifier))
    return max(roc_auc_1,roc_auc_2)

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

vdata, vlabels = load_data('Validate.csv')
vdata = scaler.transform(vdata)

flog(gt()+"data load end")

flog(tdata.shape)
flog(sum(tlabels))
flog(vdata.shape)
flog(sum(vlabels))

auc_val = 0
best_g = 0
best_c = 0
best_b = 0
for g in np.arange(0.0030, 0.0050, 0.0001):
    #for c in np.arange(1.0, 5.0, 1.0):
        #for b in np.arange(0.0, 4.0, 1.0):
            c = 3.0
            b = 0.0
            curr = run(tdata,tlabels,vdata,vlabels,g,c,b)
            if(curr > auc_val):
                auc_val = curr
                best_g = g
                best_c = c
                best_b = b
                flog("best so far "+str(auc_val)+" for "+str(g)+" and "+str(c)+" and "+str(b))
            flog("best so far "+str(auc_val)+" for "+str(best_g)+" and "+str(best_c)+" and "+str(best_b))
