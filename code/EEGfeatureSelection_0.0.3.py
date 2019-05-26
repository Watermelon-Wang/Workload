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

###### Document Decription
''' 测试scikit-learn的特征筛选算法
    *Note: Due to the compatible issue of pysam and multiprocess, I
    use a subprocess and samtools to read bam file. This should work
    on both Windows and Linux but less efficiency.

    Usage:'''

###### Version and Date
PROG_VERSION = '0.0.2'
PROG_DATE = '2019-05-05'

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
def getEEGDataFromSubject(data, subjectIndex):
    '''
    :param data: 载入的.mat文件
    :param subjectIndex: 被试的号码
    :return: 数据集和标签的字典feature
            feature['data'] -> 样本数*特征数
            feature['label'] -> 样本数*1
    '''
    feature = {}
    highData = data['AllOneChannelDiffFreqH'][0, subjectIndex]
    lowData = data['AllOneChannelDiffFreqL'][0, subjectIndex]

    sizeH = highData.shape
    sizeL = lowData.shape

    # featureSize = size[0] * size[1]
    # sampleSize = size[2]

    highData = highData.reshape(sizeH[0] * sizeH[1], -1).T
    lowData = lowData.reshape(sizeL[0] * sizeL[1], -1).T

    highLabel = np.zeros(sizeH[2])
    lowLabel = np.ones(sizeL[2])

    ## 整合数据
    feature['data'] = np.concatenate((highData, lowData), axis=0)
    feature['label'] = np.concatenate((highLabel, lowLabel), axis=0)

    ## 随机打乱
    index = np.arange(sizeH[2]+sizeL[2])
    np.random.shuffle(index)

    feature['data'] = feature['data'][index]
    feature['label'] = feature['label'][index]

    return feature

def getEyeDataFromSubject(data, subjectIndex):
    '''
    :param data: 载入的.mat文件
    :param subjectIndex: 被试的号码
    :return: 数据集和标签的字典feature
            feature['data'] -> 样本数*特征数
            feature['label'] -> 样本数*
    '''
    feature = {}
    highData = data['AllEyeFeatureH'][0, subjectIndex]
    lowData = data['AllEyeFeatureL'][0, subjectIndex]

    sizeH = highData.shape
    sizeL = lowData.shape

    highLabel = np.zeros(sizeH[1])
    lowLabel = np.ones(sizeL[1])

    highData = highData.T
    lowData = lowData.T

    ## 整合数据
    feature['data'] = np.concatenate((highData, lowData), axis=0)
    feature['label'] = np.concatenate((highLabel, lowLabel), axis=0)

    ## 随机打乱
    index = np.arange(sizeH[1] + sizeL[1])

    np.random.shuffle(index)

    feature['data'] = feature['data'][index]
    feature['label'] = feature['label'][index]

    return feature


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


def testEEGClassiferPipeline(dataFile):
    # dataFile = "E:\硕士课题\脑电程序V2\临时数据\AllOneChannelDiffFreqV003"
    ## 读取Matlab提取的特征（.mat）
    data = scipy.io.loadmat(dataFile)
    resultAll = []

    for subjectIndex in range(25):
        EEG = getEEGDataFromSubject(data, subjectIndex)
        EEG['data'] = MinMaxScaler().fit_transform(EEG['data'])
        trainData, testData, trainLabel, testLabel = \
            randomSplitData(EEG['data'], EEG['label'], scale=0.8)

        resultOne = testClassifer(trainData, testData, trainLabel, testLabel)

        resultAll.append(resultOne)
        print('第' + str(subjectIndex) + '名被试操作完成')
    print(resultAll)

    with open(r'C:\Users\Summer\Desktop\123_new.csv', 'w+') as file:
        for i in resultAll:
            for j in i:
                file.write(str(j) + ',')
            file.write('\n')
    return 0


def testEYEClassiferPipeline(dataFile):
    # dataFile = "dataFile = "E:\硕士课题\眼动程序V2\AllEyeFeatureV001.mat""
    ## 读取Matlab提取的特征（.mat）
    data = scipy.io.loadmat(dataFile)
    resultAll = []

    for subjectIndex in range(18):
        EYE = getEyeDataFromSubject(data, subjectIndex)
        EYE['data'] = MinMaxScaler().fit_transform(EYE['data'])
        trainData, testData, trainLabel, testLabel = \
            randomSplitData(EYE['data'], EYE['label'], scale=0.8)

        resultOne = testClassifer(trainData, testData, trainLabel, testLabel)

        resultAll.append(resultOne)
        print('第' + str(subjectIndex) + '名被试操作完成')
    print(resultAll)

    with open(r'C:\Users\Summer\Desktop\456.csv', 'w+') as file:
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
    dataFile = "E:\硕士课题\脑电程序V2\临时数据\AllOneChannelDiffFreqV003"
    ## 读取Matlab提取的特征（.mat）
    # data = scipy.io.loadmat(dataFile)
    # # ## 选取第三名被试
    # EYE = getEyeDataFromSubject(data, 11)
    # print(EYE['data'].shape)
    # print(EYE['label'].shape)

    ## 测试眼动数据
    # testEYEClassiferPipeline(dataFile)

    ##测试脑电数据
    testEEGClassiferPipeline(dataFile)
    print("sha qiu yue")
    # ## 把特征映射到[0,1]区间
    # EEG['data'] = MinMaxScaler().fit_transform(EEG['data'])
    #
    # #EEG['data'] = SelectKBest(chi2, k=300).fit_transform(EEG['data'], EEG['label'])
    #
    # # EEG['data'] = RFE(estimator=LogisticRegression(), n_features_to_select=300).fit_transform(EEG['data'], EEG['label'])
    #
    # trainData, testData, trainLabel, testLabel = \
    #     randomSplitData(EEG['data'], EEG['label'],scale = 0.8)
    # testClassifer(trainData, testData, trainLabel, testLabel)
    # # SVMMethod(trainData, testData, trainLabel, testLabel)
    # #
    # # XGboostMethod(trainData, testData, trainLabel, testLabel)


#################################
##
##   Start the main program.
##
#################################
if __name__ == '__main__':
    main()

################## Life is like a trip, I am just a lonely traveler. ##################
