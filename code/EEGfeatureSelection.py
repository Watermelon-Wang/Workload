#!/usr/bin/env python
## just for test pypy


###### Import Modules
import os,sys
import scipy.io
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.feature_selection import chi2

###### Document Decription
''' 测试scikit-learn的特征筛选算法
    *Note: Due to the compatible issue of pysam and multiprocess, I
    use a subprocess and samtools to read bam file. This should work
    on both Windows and Linux but less efficiency.

    Usage:'''

###### Version and Date
PROG_VERSION = '0.1.0'
PROG_DATE = '2019-01-01'

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
def getDataFromSubject(data, subjectIndex):

    '''
    :param data: 载入的.mat文件
    :param subjectIndex: 被试的号码
    :return: 数据集和标签的字典feature
            feature['data'] -> 样本数*特征数
            feature['label'] -> 样本数*1
    '''
    feature = {}
    highData = data['AllOneChannelDiffFreqH'][0, subjectIndex]
    lowData = data['AllOneChannelDiffFreqH'][0, subjectIndex]

    size = highData.shape
    featureSize = size[0]*size[1]
    sampleSize = size[2]

    highData = highData.reshape(featureSize,-1).T
    lowData = lowData.reshape(featureSize,-1).T

    highLabel = np.zeros([sampleSize,1])
    lowLabel = np.ones([sampleSize,1])

    ## 整合数据
    feature['data'] = np.concatenate((highData,lowData), axis = 0)
    feature['label'] = np.concatenate((highLabel,lowLabel), axis = 0)

    ## 随机打乱
    index = np.arange(sampleSize*2)
    np.random.shuffle(index)

    feature['data'] = feature['data'][index,:]
    feature['label'] = feature['label'][index,:]

    return feature
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
    # 读取Matlab提取的特征（.mat）
    data = scipy.io.loadmat(dataFile)
    ## 选取第三名被试
    EEG = getDataFromSubject(data, 3)
    #
    # print(EEG['data'].shape)
    # print(EEG['label'].shape)

    iris = load_iris()
    # 特征矩阵 (150, 4)
    iris.data
    # 目标向量 (150,)
    B = iris.target
    ## 对特征矩阵的列进行标准化
    # iris.data = StandardScaler().fit_transform(iris.data)
    # A = VarianceThreshold(threshold=0.001).fit_transform(iris.data)
    ## 对特征矩阵的行进行区间映射
    iris.data = MinMaxScaler().fit_transform(iris.data)
    A = SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target)
    print(B)


#################################
##
##   Start the main program.
##
#################################
if __name__ == '__main__':
    main()

################## Life is like a trip, I am just a lonely traveler. ##################
