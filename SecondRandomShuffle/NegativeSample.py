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

def MyNegativeSample():

    '''
    # 由AssociationMatrix和PositiveSampe得到PositiveSample
    '''

    # 数据
    AssociationMatrix = []
    ReadMyCsv(AssociationMatrix, "SecondRandomShuffle\AssociationMatrix.csv")
    print('AssociationMatrix[0]', AssociationMatrix[0])
    print(len(AssociationMatrix))

    PositiveSample = []
    ReadMyCsv(PositiveSample, 'SecondRandomShuffle\PositiveSample.csv')
    print(len(PositiveSample))
    print(PositiveSample[0])

    NegativeSample = []
    counterN = 0
    while counterN < len(PositiveSample):  # 随机选出一个疾病rna对，次数
        counter1 = random.randint(0, len(AssociationMatrix) - 1)
        counter2 = random.randint(0, len(AssociationMatrix[counter1]) - 1)

        flag1 = 0
        counter3 = 0
        while counter3 < len(PositiveSample):  # 正样本中是否存在
            if counter1 == PositiveSample[counter3][0] and counter2 == PositiveSample[counter3][1]:
                print('fail1')
                flag1 = 1
                break
            counter3 = counter3 + 1
        if flag1 == 1:
            continue

        flag2 = 0
        counter4 = 0
        while counter4 < len(NegativeSample):  # 在已选的负样本中没有，防止重复
            if counter1 == NegativeSample[counter4][0] and counter2 == NegativeSample[counter4][1]:
                print('fail2')
                flag2 = 1
                break
            counter4 = counter4 + 1
        if flag2 == 1:
            continue

        if (flag1 == 0 & flag2 == 0):
            Pair = []
            Pair.append(counter1)
            Pair.append(counter2)
            NegativeSample.append(Pair)         # 下三角矩阵，一定满足行 > 列

            print(counterN)
            counterN = counterN + 1

    print(len(NegativeSample))
    StorFile(NegativeSample, 'SecondRandomShuffle\\NegativeSample.csv')

    return NegativeSample

