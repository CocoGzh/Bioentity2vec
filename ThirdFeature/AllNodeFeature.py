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
import copy

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


def MyAllNodeFeature():
    AllNodeAttributeNum = []
    ReadMyCsv(AllNodeAttributeNum, 'FirstAllNodeEdge\AllNodeAttributeNum.csv')

    # 生成AllNodeManner
    counterP = 0
    while counterP < 5:
        LineEmbeddingName = 'ThirdFeature\\' + 'vec_all' + str(counterP) + '.txt'
        LineEmbedding = np.loadtxt(LineEmbeddingName, dtype=str, skiprows=1)
        print(LineEmbedding[0])

        # manner
        AllNodeMannerNum = []
        counter = 0
        while counter < len(AllNodeAttributeNum):
            pair = []
            counter1 = 0
            while counter1 < len(LineEmbedding[0]) - 1:  # 如果节点孤立，则Feature全为0，与embedding统一维度
                pair.append(0)
                counter1 = counter1 + 1
            AllNodeMannerNum.append(pair)
            counter = counter + 1

        counter = 0
        while counter < len(LineEmbedding):
            num = int(LineEmbedding[counter][0])
            AllNodeMannerNum[num] = LineEmbedding[counter][1:]
            counter = counter + 1

        print(np.array(AllNodeMannerNum).shape)
        AllNodeMannerNumName = 'ThirdFeature\\' + 'AllNodeMannerNum' + str(counterP) + '.csv'
        StorFile(AllNodeMannerNum, AllNodeMannerNumName)

        # AllNodeFeature = attribute + manner
        AllNodeFeatureNum = []
        AllNodeFeatureNum.extend(copy.deepcopy(AllNodeAttributeNum))
        counter = 0
        while counter < len(AllNodeFeatureNum):
            AllNodeFeatureNum[counter].extend(AllNodeMannerNum[counter])
            counter = counter + 1

        print(np.array(AllNodeFeatureNum).shape)
        AllNodeFeatureNumName = 'ThirdFeature\\' + 'AllNodeFeatureNum' + str(counterP) + '.csv'
        StorFile(AllNodeFeatureNum, AllNodeFeatureNumName)

        counterP = counterP + 1
    return
