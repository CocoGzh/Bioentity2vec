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

from SecondRandomShuffle.AssociationMatrix import MyAssociationMatrix
MyAssociationMatrix = MyAssociationMatrix()

from SecondRandomShuffle.RandomListPositiveSample import MyRandomListPositiveSample
PositiveSample, RandomListGroup = MyRandomListPositiveSample()

from SecondRandomShuffle.NegativeSample import MyNegativeSample
NegativeSample = MyNegativeSample()

from SecondRandomShuffle.Segmentation import MySegmentation
Segmentation = MySegmentation()