#!/usr/bin/env python
## just for test pypy


###### Import Modules
import os, sys
import scipy.io
import numpy as np
import time

###### Document Decription
''' 测试mne算法
    *Note: Due to the compatible issue of pysam and multiprocess, I
    use a subprocess and samtools to read bam file. This should work
    on both Windows and Linux but less efficiency.

    Usage:'''

###### Version and Date
PROG_VERSION = '0.0.1'
PROG_DATE = '2019-05-24'

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
    lowData = data['AllOneChannelDiffFreqH'][0, subjectIndex]


    feature['H'] = feature['data'][index]
    feature['L'] = feature['label'][index]

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
    data = scipy.io.loadmat(dataFile)
    getEEGDataFromSubject(data, 3)
    # Generate some random data: 10 epochs, 5 channels, 2 seconds per epoch
    sfreq = 100
    data = np.random.randn(10, 5, sfreq * 2)

    # Initialize an info structure
    info = mne.create_info(
        ch_names=['MEG1', 'MEG2', 'EEG1', 'EEG2', 'EOG'],
        ch_types=['grad', 'grad', 'eeg', 'eeg', 'eog'],
        sfreq=sfreq
    )


#################################
##
##   Start the main program.
##
#################################
if __name__ == '__main__':
    main()

################## Life is like a trip, I am just a lonely traveler. ##################
