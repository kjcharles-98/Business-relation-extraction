from sklearn import svm
from sklearn import metrics as m
import spacy
import numpy as np
from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB
import pandas as pd
from sklearn.ensemble import VotingClassifier


def allfloat(arr):
    re_list = []
    for i in arr:
        lst = []
        for j in i:
            element = float(pd.to_numeric(j))
            lst.append(element)
        re_list.append(lst)
    return np.array(re_list)

def voteclf(X,y):
    lr = LogisticRegression(C = 100,class_weight = 'balanced', dual=False, max_iter = 10000, multi_class = 'ovr',penalty = 'l1', solver='liblinear', tol=0.001)
    svm = LinearSVC(C=1, class_weight='balanced', dual=False, max_iter=10000, multi_class='ovr', tol=0.0001)
    #nb = GaussianNB()
    vote = VotingClassifier(estimators=[('lr', lr), ('svm', svm)], voting='hard')  # 无权重投票
    for clf, label in zip([lr, svm, vote], ['Logistic Regression', 'LinearSVC', 'Ensemble']):
        scores = cross_val_score(clf,X,y,cv=5, scoring='accuracy')
        print("Precision: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    return vote



def train_svm(xs,ys,clf_name):

    #parameter for svm
    para_dict = {'svm':[{'dual': [False],'tol':[1e-3, 1e-4],'C': [1, 10, 100, 1000],'multi_class': ['ovr'],'class_weight':["balanced"],'max_iter':[10000]},{'loss': ['squared_hinge','hinge'],'dual': [True],'tol':[1e-3, 1e-4],'C': [1, 10, 100, 1000],'multi_class': ['ovr'],'class_weight':["balanced"],'max_iter':[10000]}],
                'lr':{'solver':['liblinear'],'penalty':['l1','l2'],'dual': [False],'tol':[1e-3, 1e-4],'C': [1, 10, 100, 1000],'multi_class': ['ovr'],'class_weight':["balanced"],'max_iter':[10000]},
                'tree':{'criterion':['entropy','gini'], 'splitter':['best','random'],'max_depth':[30,40,50,60,70],'min_samples_leaf':[2,3,4,5], 'min_samples_split':[2,3,4,5,6,7,8,9,10],'class_weight':['balanced']},
                'nb':{'fit_prior':[True,False]},
                'vote':{}
                }


    clf_dict = {'svm':LinearSVC(), 'lr':LogisticRegression(), 'tree':DecisionTreeClassifier(),'nb':MultinomialNB()}
    if clf_name == 'vote':
        clf_dict[clf_name] = voteclf(xs,ys)


    tuned_parameters = para_dict[clf_name]
    clf = clf_dict[clf_name]

    classifier = GridSearchCV(clf, tuned_parameters,cv=10,scoring='accuracy',refit=True)



    print("\n\n\n\n\n----------START----------\n\n\n\n\n")

    #ratio = len(ys)-int(0.2*len(ys))
    #x_train = xs[:ratio,:]
    #x_test = xs[ratio:,:]
    #y_train = ys[:ratio]
    #y_test = ys[ratio:]
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=.3,random_state=0)

    print(np.array(x_train).shape)
    print(np.array(y_train).shape)
    print(np.array(x_test).shape)
    print(np.array(y_test).shape)
    classifier.fit(x_train,y_train)

    print("best_params")
    print(classifier.best_params_)
    pred = classifier.predict(x_test)
    print("train precision: ",format(classifier.score(x_train,y_train),'.2f'))
    print("test precison: ",format(classifier.score(x_test,y_test),'.2f'))


    print('Test recall score:{:.3f}'.format(m.recall_score(y_test,pred,average="micro")))
    print('Test f measure score:{:.3f}'.format(m.f1_score(y_test,pred,average="micro")))
    print('Test precision score:{:.3f}'.format(m.precision_score(y_test,pred,average="micro")))
    c_matrix = m.confusion_matrix(y_test, pred)
    print(c_matrix)
    print('predict_result:',pred)





if __name__ == '__main__':
    y_x = np.loadtxt("tag_combined/alltag_combined_features.txt",delimiter=",",dtype=float)
    ys = y_x[:,0]
    xs = y_x[:,1:]
    print("shape")
    print(xs.shape)
    print(len(ys))
    print(ys)
    print(type(ys))
    #ratio = len(ys)-int(0.1*len(ys))
    #print(ratio)
    train_svm(xs,ys,'svm')
    print("0.2")
