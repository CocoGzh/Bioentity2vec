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

def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:          # 注意表头
        SaveList.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

# 数据
# LD
AllNode = []
ReadMyCsv(AllNode, "AllNode.csv")
print('AllNode[0]', AllNode[0])
AllNodeAttribute = []
ReadMyCsv(AllNodeAttribute, "AllNodeAttribute.csv")
print('AllNodeAttribute[0]', AllNodeAttribute[0])

num = 0
counter = 0
while counter < len(AllNode):
    if AllNode[counter][0] == AllNodeAttribute[counter][0]:
        num = num + 1
        counter = counter + 1
        continue
    counter = counter + 1

print(num)
