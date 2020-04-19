from sklearn import svm
from sklearn import metrics as m
import spacy
import numpy as np
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,classification_report,auc
from sklearn.preprocessing import label_binarize
from sklearn import model_selection,metrics
from scipy import interp
from itertools import cycle
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB
from sklearn.ensemble import VotingClassifier
from operator import truediv
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


def showFeatureImportance(model):
    #FEATURE IMPORTANCE
    # Get Feature Importance from the classifier
    feature_importance = model.feature_importances_

    # Normalize The Features
    feature_importance = 100.0 * (feature_importance / Feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    #plot relative feature importance
    plt.figure(figsize=(12, 12))
    plt.barh(pos, feature_importance[sorted_idx], align='center', color='#7A68A6')
    plt.yticks(pos, np.asanyarray(X_cols)[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance')
    plt.show()

#{'C': 10, 'class_weight': 'balanced', 'dual': False, 'max_iter': 10000, 'multi_class': 'ovr', 'tol': 0.0001}
def train_svm(X,y,clfname):

    if 1==1:
        clf_dict = {'verb_svm':svm.LinearSVC(C=1000,class_weight='balanced',dual=False,max_iter=10000,multi_class='ovr',tol=0.001),
                   'verb_lr':LogisticRegression(C=100, class_weight='balanced',dual=False, max_iter=10000,multi_class='ovr',penalty='l2',solver='liblinear',tol=0.001),
                   'tree':DecisionTreeClassifier(criterion='gini',class_weight='balanced',max_depth=50, min_samples_leaf=2,min_samples_split=3,splitter='random'),
                   'multi':MultinomialNB(fit_prior=False),
                   'tf_svm':LinearSVC(C=1,class_weight='balanced',dual=False,max_iter=10000,multi_class='ovr',tol=0.001),
                   'tf_lr':LogisticRegression(C=10, class_weight='balanced',dual=False, max_iter=10000,multi_class='ovr',penalty='l2',solver='liblinear',tol=0.001),
                   'com_lr':LogisticRegression(C=10, class_weight='balanced',dual=False, max_iter=10000,multi_class='ovr',penalty='l1',solver='liblinear',tol=0.0001),
                   'tag_lr':LogisticRegression(C=10, class_weight='balanced',dual=False, max_iter=10000,multi_class='ovr',penalty='l1',solver='liblinear',tol=0.001),
                   'tag_svm':LinearSVC(C=1,class_weight='balanced',dual=False,max_iter=10000,multi_class='ovr',tol=0.001)
                   }


        # 设置种类
        n_classes = 12

        # 训练模型并预测
        random_state = np.random.RandomState(0)
        n_samples, n_features = X.shape

        # shuffle and split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,random_state=0)


        classifier=clf_dict[clfname]
        l = ['svm','lr','tf_svm','tf_lr','tag_svm','tag_lr','com_lr']
        if clfname in l :
            y_score = classifier.fit(X_train, y_train).decision_function(X_test)
        else:
            model = classifier.fit(X_train, y_train)
            probs = model.predict_proba(X_test)
            y_score = probs

        print("train")
        print(X_train)
        print(y_train)
        model = classifier.fit(X_train, y_train)

        print("train precision: ",format(classifier.score(X_train,y_train),'.2f'))
        print("test precison: ",format(classifier.score(X_test,y_test),'.2f'))
        pred = classifier.predict(X_test)
        print(y_test.shape)
        print("score")
        print(y_score.shape)

        #showFeatureImportance(classifier)

        print('Micro: Test recall score:{:.3f}'.format(m.recall_score(y_test,pred,average="micro")))
        print('Micro: Test f measure score:{:.3f}'.format(m.f1_score(y_test,pred,average="micro")))
        print('Micro: precision score:{:.3f}'.format(m.precision_score(y_test,pred,average="micro")))
        print("\n-----------------\n")

        print('Macro: Test recall score:{:.3f}'.format(m.recall_score(y_test,pred,average="macro")))
        print('Macro: Test f measure score:{:.3f}'.format(m.f1_score(y_test,pred,average="macro")))
        print('Macro: Test precision score:{:.3f}'.format(m.precision_score(y_test,pred,average="macro")))
        print("\n-----------------\n")

        print('Weighted: Test recall score:{:.3f}'.format(m.recall_score(y_test,pred,average="weighted")))
        print('Weighted: Test f measure score:{:.3f}'.format(m.f1_score(y_test,pred,average="weighted")))
        print('Weighted: Test precision score:{:.3f}'.format(m.precision_score(y_test,pred,average="weighted")))
        print("\n-----------------\n")

        con_m = m.confusion_matrix(y_test, pred)
        diag = np.diag(con_m)
        raw_sum = np.sum(con_m,axis=1)
        each_acc = np.nan_to_num(truediv(diag,raw_sum))
        print("--------------------")
        print(each_acc)
        print("--------------------")


        # 计算每一类的ROC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        y_test = label_binarize(y_test, classes=[0,1,2,3,4,5,6,7,8,9,10,11,12])
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area（方法二）
        #fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        #roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area（方法一）
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        lw=2
        plt.figure()
        #plt.plot(fpr["micro"], tpr["micro"],
        #         label='micro-average ROC curve (area = {0:0.2f})'
        #               ''.format(roc_auc["micro"]),
        #         color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()
















if __name__ == '__main__':
    #xs,ys = pre.feature_extraction()
    y_x = np.loadtxt("alltagdep_combined_features.txt",delimiter=",",dtype=float)
    ys = y_x[:,0]
    xs = y_x[:,1:]
    print("shape")
    print(xs.shape)
    print(len(ys))
    print(ys)
    print(type(ys))
    ratio = len(ys)-int(0.1*len(ys))
    #print(ratio)
    #train_svm(xs[:ratio,:],xs[ratio:,:],ys[:ratio],ys[ratio:])
    train_svm(xs,ys,'tf_lr')
    print("0.2")
