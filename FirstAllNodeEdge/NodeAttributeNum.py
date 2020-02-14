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
    for row in csv_reader:          # 注意表头
        SaveList.append(row)
    return

def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        for i in range(len(row)):       # 转换数据类型
            row[i] = float(row[i])
        SaveList.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


# 数据
# 节点
def MyNodeAttributeNum():
    AllNode = []
    ReadMyCsv(AllNode, "FirstAllNodeEdge\AllNode.csv")
    print('len(AllNode)', len(AllNode))
    print('AllNode[0]', AllNode[0])

    AllNodeNum = []
    counter = 0
    while counter < len(AllNode):
        pair = []
        pair.append(counter)
        AllNodeNum.append(pair)
        counter = counter + 1
    print('AllNodeNum[0]', AllNodeNum[0])

    StorFile(AllNodeNum, 'FirstAllNodeEdge\AllNodeNum.csv')

    AllNodeAttribute = []
    ReadMyCsv(AllNodeAttribute, "FirstAllNodeEdge\AllNodeAttribute.csv")
    print('len(AllNodeAttribute)', len(AllNodeAttribute))
    print('AllNodeAttribute[0]', AllNodeAttribute[0])

    AllNodeAttributeNum = []
    counter = 0
    while counter < len(AllNodeAttribute):
        AllNodeAttributeNum.append(AllNodeAttribute[counter][1:])
        counter = counter + 1
    print('AllNodeAttributeNum[0]', AllNodeAttributeNum[0])
    StorFile(AllNodeAttributeNum, 'FirstAllNodeEdge\AllNodeAttributeNum.csv')

    return AllNodeNum, AllNodeAttributeNum
