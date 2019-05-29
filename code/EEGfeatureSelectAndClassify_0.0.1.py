#!/usr/bin/env python
## just for test pypy


###### Import Modules
import os, sys
import scipy.io
import numpy as np
# import xgboost as xgb
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, LeaveOneOut, LeavePOut, ShuffleSplit  # 交叉验证所需的子集划分方法
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit  # 分层分割
# from xgboost import plot_importance
from matplotlib import pyplot as plt
# 计算分类正确率
from sklearn.metrics import accuracy_score

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

###### Document Decription
''' mental workload project
    *对脑电数据特征进行筛选和分类

    Usage:'''

###### Version and Date
PROG_VERSION = '0.0.1'
PROG_DATE = '2019-05-25'

###### Usage
usage = '''

     Version %s  by Wang Tao  %s

     Usage: %s <> >STDOUT
''' % (PROG_VERSION, PROG_DATE, os.path.basename(sys.argv[0]))


######## Global Variable


#######################################################################
############################  BEGIN Class  ############################
#######################################################################


##########################################################################
############################  BEGIN Function  ############################
##########################################################################

def printRunTime(func):
    def wrapper(*args, **kw):
        localTime = time.time()
        res = func(*args, **kw)
        print('current Function [%s] run time is %.2f' % (func.__name__ ,time.time() - localTime) )
        return res
    return wrapper

def getEEGDataFromSubject(dataPath, subjectIndex, outType = 1):
    '''
    :param dataPath:
    :param subjectIndex:
    :param outType:
        outType为1或其他时：输出的特征矩阵为，L1、H1 任务态的脑电特征
        outType为2时：输出的特征矩阵为，L1、L2、H1、H2 任务态的脑电特征
        outType为3时：输出的特征矩阵为，L1的Rest\work 的脑电特征
        outType为4时：输出的特征矩阵为，H1的Rest\work 的脑电特征
    :return:
    '''
    outType = int(outType)

    data = scipy.io.loadmat(dataPath)

    # data1:一般为静息态或者低脑力负荷
    # data1:一般为任务态或者高脑力负荷
    if outType == 2:
        classData1 = data['featureLWall'][0, subjectIndex]
        classData2 = data['featureHWall'][0, subjectIndex]
    if outType == 3:
        classData1 = data['featureL1R'][0, subjectIndex]
        classData2 = data['featureL1W'][0, subjectIndex]
    if outType == 4:
        classData1 = data['featureH1R'][0, subjectIndex]
        classData2 = data['featureH1W'][0, subjectIndex]
    else:
        classData1 = data['featureL1W'][0, subjectIndex]
        classData2 = data['featureH1W'][0, subjectIndex]

    size1 = classData1.shape
    size2 = classData2.shape
    # eg.(60, 39, 144):size[0],size[1],size[2]

    classData1 = classData1.reshape(size1[0] * size1[1], -1).T
    classData2 = classData2.reshape(size2[0] * size2[1], -1).T

    classLabel1 = np.zeros(size1[2])
    classLabel2 = np.ones(size2[2])

    ## 整合数据
    feature = np.concatenate((classData1, classData2), axis=0)
    label = np.concatenate((classLabel1, classLabel2), axis=0)

    ## 随机打乱
    index = np.arange(size1[2]+size2[2])
    np.random.shuffle(index)

    feature = feature[index]
    label = label[index]

    return feature,label

# @printRunTime
def EEG3trailPredict1Trail(dataPath, subjectIndex):

    data = scipy.io.loadmat(dataPath)

    trainData = {}
    testData = {}
    trainLabel = {}
    testLabel = {}

    mapIndex = {
        0: [1,2,3],
        1: [0,2,3],
        2: [0,1,3],
        3: [0,1,2]
    }

    classData = {}
    classLabel = {}

    classData[0] = data['featureL1W'][0, subjectIndex]
    classData[1] = data['featureL2W'][0, subjectIndex]
    classData[2] = data['featureH1W'][0, subjectIndex]
    classData[3] = data['featureH2W'][0, subjectIndex]

    size0 = classData[0].shape
    size1 = classData[1].shape
    size2 = classData[2].shape
    size3 = classData[3].shape
    # eg.(60, 39, 144):size[0],size[1],size[2]

    classData[0] = classData[0].reshape(size0[0] * size0[1], -1).T
    classData[1] = classData[1].reshape(size1[0] * size1[1], -1).T
    classData[2] = classData[2].reshape(size2[0] * size2[1], -1).T
    classData[3] = classData[3].reshape(size3[0] * size3[1], -1).T

    classLabel[0] = np.zeros(size0[2])
    classLabel[1] = np.zeros(size1[2])
    classLabel[2] = np.ones(size2[2])
    classLabel[3] = np.ones(size3[2])

    resultOne = {}

    # for classIndex in range(1):
    #
    #     trainData = np.concatenate((classData[mapIndex[classIndex][0]], classData[mapIndex[classIndex][1]], classData[mapIndex[classIndex][2]]), axis=0)
    #     trainLabel = np.concatenate((classLabel[mapIndex[classIndex][0]], classLabel[mapIndex[classIndex][1]], classLabel[mapIndex[classIndex][2]]), axis=0)
    #     testData = classData[classIndex]
    #     testLabel = classLabel[classIndex]
    #
    #     trainData = MinMaxScaler().fit_transform(trainData)
    #     testData = MinMaxScaler().fit_transform(testData)
    #
    #     # resultOne[classIndex] = testClassifer(trainData, testData, trainLabel, testLabel)
    #     A = testClassifer(trainData, testData, trainLabel, testLabel)
    #
    # resultOneMean = A

    for classIndex in range(4):

        trainData = np.concatenate((classData[mapIndex[classIndex][0]], classData[mapIndex[classIndex][1]], classData[mapIndex[classIndex][2]]), axis=0)
        trainLabel = np.concatenate((classLabel[mapIndex[classIndex][0]], classLabel[mapIndex[classIndex][1]], classLabel[mapIndex[classIndex][2]]), axis=0)
        testData = classData[classIndex]
        testLabel = classLabel[classIndex]

        trainData = MinMaxScaler().fit_transform(trainData)
        testData = MinMaxScaler().fit_transform(testData)

        resultOne[classIndex] = testClassifer(trainData, testData, trainLabel, testLabel)

    resultOneMean = resultOne[0] + resultOne[1] + resultOne[2] + resultOne[3]

    return resultOneMean

def EEG3trailPredict1TrailPipeline(dataPath):

    resultAll = []

    for subjectIndex in range(18):

        resultOne = EEG3trailPredict1Trail(dataPath, subjectIndex)
        resultAll.append(resultOne)
        print('第' + str(subjectIndex) + '名被试操作完成')

    print(resultAll)

    with open(r'C:\Users\WangTao\Desktop\abc123456.csv', 'w+') as file:
        for i in resultAll:
            for j in i:
                file.write(str(j) + ',')
            file.write('\n')
    return 0


def StratifiedKFoldMethod(data, label):
    '''
    :param data: 特征矩阵
    :param label: 标签
    :return: 训练集数据和标签、测试集数据和标签
    '''
    ##交叉验证,分层K折
    # Result = {}
    skf = StratifiedKFold(n_splits=3)

    for trainIndex, testIndex in skf.split(data, label):
        print("分层K折划分：%s %s" % (trainIndex.shape, testIndex.shape))
        break

    # Result['trainData'], Result['testData'] = data[trainIndex], data[testIndex]
    # Result['trainLabel'], Result['testLabel'] = label[trainIndex], label[testIndex]

    trainData, testData = data[trainIndex], data[testIndex]
    trainLabel, testLabel = label[trainIndex], label[testIndex]
    # return Result
    return trainData, testData, trainLabel, testLabel


def randomSplitData(data, label, scale=0.8):
    '''
    :param data: 特征矩阵
    :param label: 标签
    :return: 训练集数据和标签、测试集数据和标签
    random_state：是随机数的种子。
    比如每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
    '''
    from sklearn.model_selection import train_test_split
    trainData, testData, trainLabel, testLabel = \
        train_test_split(data, label, random_state=1, train_size=scale)
    return trainData, testData, trainLabel, testLabel

def testClassifer(trainData, testData, trainLabel, testLabel):
    accList = []

    classifiersName = ['NB', 'KNN', 'LR', 'RF', 'DT', 'GBDT', 'NuSVC', 'LSVC', 'SVC']

    classifiers = {'NB': naiveBayesMethod,
                   'KNN': KNNMethod,
                   'LR': logisticRegressionMethod,
                   'RF': randomForestMethod,
                   'DT': decisionTreeMethod,
                   'GBDT': gradientBoostingMethod,
                   'SVMCV': SVMCrossValidation,
                   'NuSVC': NuSVCMethod,
                   'LSVC': linearSVCMethod,
                   'SVC': SVCMethod
                   # 'XGB': XGboostMethod
                   }
    for classifier in classifiersName:
        info = classifiers[classifier](trainData, testData, trainLabel, testLabel)
        # print('******************* %s ********************' % info['name'])
        # print('training took %fs!' % info['time'])
        # print('accuracy: %.2f%%' % (100 * info['accuracy']))

        accList.append(info['accuracy'])

    return accList


def testEEGClassiferPipeline(dataPath):

    resultAll = []

    for subjectIndex in range(18):
        feature,label = getEEGDataFromSubject(dataPath, subjectIndex,2)
        feature = MinMaxScaler().fit_transform(feature)
        trainData, testData, trainLabel, testLabel = \
            randomSplitData(feature,label, scale=0.8)

        resultOne = testClassifer(trainData, testData, trainLabel, testLabel)

        resultAll.append(resultOne)
        print('第' + str(subjectIndex) + '名被试操作完成')
    print(resultAll)

    with open(r'C:\Users\WangTao\Desktop\abc123.csv', 'w+') as file:
        for i in resultAll:
            for j in i:
                file.write(str(j) + ',')
            file.write('\n')
    return 0


def testBetweenSubjestP300ClassiferPipeline(dataFile):
    pass

##########################################################################
###############################  分类算法  ################################
##########################################################################

# def XGboostMethod(trainData, testData, trainLabel, testLabel):
#     info = {
#         'name': 'XGboostMethod',
#         'accuracy': 0,
#         'time': 0,
#         'remark': ''
#     }
#     startTime = time.time()
#     from sklearn.metrics import accuracy_score
#     ## 读取数据
#     dtrain = xgb.DMatrix(trainData, trainLabel)
#     dtest = xgb.DMatrix(testData, testLabel)
#     ## 存成二进制文件，方便调用
#     # dtrain.save_binary("train.buffer")
#     # dtest.save_binary("test.buffer")
#
#     # dtrain = xgb.DMatrix('train.buffer')
#     # dtest = xgb.DMatrix('test.buffer')
#
#     ## 设置参数
#     param = {
#         'booster': 'gbtree',
#         'objective': 'multi:softmax',  # 多分类的问题
#         'num_class': 2,  # 类别数，与 multisoftmax 并用
#         'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
#         'max_depth': 12,  # 构建树的深度，越大越容易过拟合
#         'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#         'subsample': 0.7,  # 随机采样训练样本
#         'colsample_bytree': 0.7,  # 生成树时进行的列采样
#         'min_child_weight': 3,
#         'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
#         'eta': 0.3,  # 如同学习率
#         'seed': 1000,
#         'nthread': 4,  # cpu 线程数
#     }
#     # param = {'max_depth':2, 'eta':1, 'silent':0, 'objective':'binary:logistic' }
#     numRound = 2
#
#     bst = xgb.train(param, dtrain, numRound)
#
#     trainPredictions = [round(value) for value in bst.predict(dtrain)]
#     testPredictions = [round(value) for value in bst.predict(dtest)]
#
#     trainLabelResult = dtrain.get_label()  # 值为输入数据的第一行
#     testLabelResult = dtest.get_label()
#
#     trainAccuracy = accuracy_score(trainLabelResult, trainPredictions)
#     testAccuracy = accuracy_score(testLabelResult, testPredictions)
#
#     # print ("Train Accuary: %.2f%%" % (trainAccuracy * 100.0))
#     # print("Test Accuracy: %.2f%%" % (testAccuracy * 100.0))
#
#     # 显示重要特征
#     # plot_importance(bst)
#     # plt.show()
#     info['time'] = time.time() - startTime
#     info['accuracy'] = trainAccuracy
#
#     return info


def LDAMethod(trainData, testData, trainLabel, testLabel):
    pass


def SVCMethod(trainData, testData, trainLabel, testLabel):
    info = {
        'name': 'SVCMethod',
        'accuracy': 0,
        'time': 0,
        'remark': ''
    }
    startTime = time.time()

    from sklearn import svm

    clf = svm.SVC()

    clf.fit(trainData, trainLabel)
    labelPred = clf.predict(testData)
    testAccuracy = accuracy_score(testLabel, labelPred)
    # print("SVM Test Accuracy: %.2f%%" % (testAccuracy * 100.0))
    info['time'] = time.time() - startTime
    info['accuracy'] = testAccuracy

    return info


def linearSVCMethod(trainData, testData, trainLabel, testLabel):
    info = {
        'name': 'linearSVCMethod',
        'accuracy': 0,
        'time': 0,
        'remark': ''
    }
    startTime = time.time()

    from sklearn.svm import LinearSVC

    clf = LinearSVC()

    clf.fit(trainData, trainLabel)
    labelPred = clf.predict(testData)
    testAccuracy = accuracy_score(testLabel, labelPred)
    # print("SVM Test Accuracy: %.2f%%" % (testAccuracy * 100.0))
    info['time'] = time.time() - startTime
    info['accuracy'] = testAccuracy

    return info


def NuSVCMethod(trainData, testData, trainLabel, testLabel):
    info = {
        'name': 'NuSVCMethod',
        'accuracy': 0,
        'time': 0,
        'remark': ''
    }
    startTime = time.time()

    from sklearn.svm import NuSVC

    clf = NuSVC()

    clf.fit(trainData, trainLabel)
    labelPred = clf.predict(testData)
    testAccuracy = accuracy_score(testLabel, labelPred)
    # print("SVM Test Accuracy: %.2f%%" % (testAccuracy * 100.0))
    info['time'] = time.time() - startTime
    info['accuracy'] = testAccuracy

    return info


def naiveBayesMethod(trainData, testData, trainLabel, testLabel):
    info = {
        'name': 'naiveBayesMethod',
        'accuracy': 0,
        'time': 0,
        'remark': ''
    }
    startTime = time.time()

    from sklearn.naive_bayes import MultinomialNB

    model = MultinomialNB(alpha=0.01)
    model.fit(trainData, trainLabel)
    labelPred = model.predict(testData)
    testAccuracy = accuracy_score(testLabel, labelPred)

    info['time'] = time.time() - startTime
    info['accuracy'] = testAccuracy

    return info


def KNNMethod(trainData, testData, trainLabel, testLabel):
    info = {
        'name': 'KNNMethod',
        'accuracy': 0,
        'time': 0,
        'remark': ''
    }
    startTime = time.time()

    from sklearn.neighbors import KNeighborsClassifier

    model = KNeighborsClassifier()
    model.fit(trainData, trainLabel)
    labelPred = model.predict(testData)
    testAccuracy = accuracy_score(testLabel, labelPred)

    info['time'] = time.time() - startTime
    info['accuracy'] = testAccuracy

    return info


def logisticRegressionMethod(trainData, testData, trainLabel, testLabel):
    info = {
        'name': 'logisticRegressionMethod',
        'accuracy': 0,
        'time': 0,
        'remark': ''
    }
    startTime = time.time()

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(penalty='l2')
    model.fit(trainData, trainLabel)
    labelPred = model.predict(testData)
    testAccuracy = accuracy_score(testLabel, labelPred)

    info['time'] = time.time() - startTime
    info['accuracy'] = testAccuracy

    return info


def randomForestMethod(trainData, testData, trainLabel, testLabel):
    info = {
        'name': 'randomForestMethod',
        'accuracy': 0,
        'time': 0,
        'remark': ''
    }
    startTime = time.time()

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=8)
    model.fit(trainData, trainLabel)
    labelPred = model.predict(testData)
    testAccuracy = accuracy_score(testLabel, labelPred)

    info['time'] = time.time() - startTime
    info['accuracy'] = testAccuracy

    return info


def decisionTreeMethod(trainData, testData, trainLabel, testLabel):
    info = {
        'name': 'decisionTreeMethod',
        'accuracy': 0,
        'time': 0,
        'remark': ''
    }
    startTime = time.time()

    from sklearn import tree

    model = tree.DecisionTreeClassifier()
    model.fit(trainData, trainLabel)
    labelPred = model.predict(testData)
    testAccuracy = accuracy_score(testLabel, labelPred)

    info['time'] = time.time() - startTime
    info['accuracy'] = testAccuracy

    return info


def gradientBoostingMethod(trainData, testData, trainLabel, testLabel):
    info = {
        'name': 'gradientBoostingMethod',
        'accuracy': 0,
        'time': 0,
        'remark': ''
    }
    startTime = time.time()

    from sklearn.ensemble import GradientBoostingClassifier

    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(trainData, trainLabel)
    labelPred = model.predict(testData)
    testAccuracy = accuracy_score(testLabel, labelPred)

    info['time'] = time.time() - startTime
    info['accuracy'] = testAccuracy

    return info


def SVMCrossValidation(trainData, testData, trainLabel, testLabel):
    info = {
        'name': 'SVMCrossValidation',
        'accuracy': 0,
        'time': 0,
        'remark': ''
    }
    startTime = time.time()

    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC

    model = SVC(kernel='rbf', probability=True)
    paramGrid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    gridSearch = GridSearchCV(model, paramGrid, n_jobs=1, verbose=1)
    gridSearch.fit(trainData, trainLabel)
    bestParameters = gridSearch.best_estimator_.get_params()

    model = SVC(kernel='rbf', C=bestParameters['C'], gamma=bestParameters['gamma'], probability=True)

    model.fit(trainData, trainLabel)
    labelPred = model.predict(testData)
    testAccuracy = accuracy_score(testLabel, labelPred)
    info['time'] = time.time() - startTime
    info['accuracy'] = testAccuracy

    return info


######################################################################
############################  BEGIN Main  ############################
######################################################################
#################################
##
##   Main function of program.
##
#################################
def main():
    ######################### Phrase parameters #########################

    ############################# Main Body #############################
    dataPath = r"F:\workloadProject\EEGresult\EEGmatlabResult\feature_L1L2H1H2_RW_day1_2s_1_40Hz"
    # testEEGClassiferPipeline(dataPath)
    EEG3trailPredict1TrailPipeline(dataPath)
    # B = EEG3trailPredict1Trail(dataPath, 4)
    # print(EEG3trailPredict1Trail(dataPath, 4))

#################################
##
##   Start the main program.
##
#################################
if __name__ == '__main__':
    main()

################## Life is like a trip, I am just a lonely traveler. ##################
