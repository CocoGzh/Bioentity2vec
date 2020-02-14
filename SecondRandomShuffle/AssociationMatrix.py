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


def MyAssociationMatrix():
    # 数据
    AllNodeNum = []
    ReadMyCsv(AllNodeNum, "FirstAllNodeEdge\AllNodeNum.csv")
    print('AllNode[0]', AllNodeNum[0])
    print(len(AllNodeNum))

    AllEdgeNum = []
    ReadMyCsv(AllEdgeNum, "FirstAllNodeEdge\AllEdgeNum.csv")
    print('AllEdge[0]', AllEdgeNum[0])
    print(len(AllEdgeNum))

    # 下三角矩阵
    AssociationMatrix = []
    counter = 0
    while counter < len(AllNodeNum):
        Row = []
        counter1 = 0
        while counter1 <= counter:
            Row.append(0)
            counter1 = counter1 + 1
        AssociationMatrix.append(Row)
        print(counter)
        counter = counter + 1

    counter = 0
    while counter < len(AllEdgeNum):
        PairA = AllEdgeNum[counter][0]
        PairB = AllEdgeNum[counter][1]

        counter1 = 0
        while counter1 < len(AllNodeNum):
            if int(PairA) == int(AllNodeNum[counter1][0]):
                counterA = counter1
                break
            counter1 = counter1 + 1

        counter2 = 0
        while counter2 < len(AllNodeNum):
            if int(PairB) == int(AllNodeNum[counter2][0]):
                counterB = counter2
                break
            counter2 = counter2 + 1


        if counterA < counterB:
            temp = counterB
            counterB = counterA
            counterA = temp

        AssociationMatrix[counterA][counterB] = 1

        print(counter)
        counter = counter + 1

    print(len(AssociationMatrix))
    StorFile(AssociationMatrix, 'SecondRandomShuffle\AssociationMatrix.csv')



    return AssociationMatrix