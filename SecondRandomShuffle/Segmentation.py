from numpy import *
import numpy as np
import random
import math
import os
import time
import pandas as pd
import csv
import math
import random

# 定义函数
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return

def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 0
        while counter < len(row):
            row[counter] = int(row[counter])      # 转换数据类型
            counter = counter + 1
        SaveList.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

def MySegmentation():
    '''
    PositiveSample/NegativeSample都是乱序
    由NewRandomList每折的长度生成Train/Test并保存，送至embedding训练
    :return:
    '''
    PositiveSample = []
    ReadMyCsv(PositiveSample, "SecondRandomShuffle\PositiveSample.csv")
    print('PositiveSample[0]', PositiveSample[0])
    print(len(PositiveSample))

    # NegativeSample = []
    # ReadMyCsv(NegativeSample, "SecondRandomShuffle\\NegativeSample.csv")
    # print('NegativeSample[0]', NegativeSample[0])
    # print(len(NegativeSample))

    RandomListGroup = []
    ReadMyCsv(RandomListGroup, "SecondRandomShuffle\RandomListGroup.csv")
    print('NewRandomList[0]', RandomListGroup[0])
    print(len(RandomListGroup))

    # 五折的训练集和测试集
    counter = 0
    while counter < len(RandomListGroup):

        Num = 0
        TestListPair = []
        TrainListPair = []
        counter2 = 0
        while counter2 < len(RandomListGroup):
            if counter2 == counter:
                TestListPair.extend(PositiveSample[Num:Num + len(RandomListGroup[counter2])])
            if counter2 != counter:
                TrainListPair.extend(PositiveSample[Num:Num + len(RandomListGroup[counter2])])
            Num = Num + len(RandomListGroup[counter2])
            counter2 = counter2 + 1

        TestName = 'SecondRandomShuffle\\' + 'TestName' + str(counter) + '.csv'
        StorFile(TestListPair, TestName)
        TrainName = 'SecondRandomShuffle\\' + 'TrainName' + str(counter) + '.csv'
        StorFile(TrainListPair, TrainName)

        counter = counter + 1

    return




