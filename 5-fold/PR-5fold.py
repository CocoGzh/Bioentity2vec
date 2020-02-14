import os
import time
import numpy as np
import pandas as pd
import csv
import math
import random
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
from sklearn.model_selection import KFold,LeaveOneOut,LeavePOut,ShuffleSplit

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from inspect import signature


# 定义函数
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        for i in range(len(row)):
            row[i] = float(row[i])
        SaveList.append(row)
    return

def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

def MyEnlarge(x0, y0, width, height, x1, y1, times, mean_fpr, mean_tpr, thickness=1, color = 'blue'):
    def MyFrame(x0, y0, width, height):
        import matplotlib.pyplot as plt
        import numpy as np

        x1 = np.linspace(x0, x0, num=20)  # 生成列的横坐标，横坐标都是x0，纵坐标变化
        y1 = np.linspace(y0, y0, num=20)
        xk = np.linspace(x0, x0 + width, num=20)
        yk = np.linspace(y0, y0 + height, num=20)

        xkn = []
        ykn = []
        counter = 0
        while counter < 20:
            xkn.append(x1[counter] + width)
            ykn.append(y1[counter] + height)
            counter = counter + 1

        plt.plot(x1, yk, color='k', linestyle=':', lw=1, alpha=1)  # 左
        plt.plot(xk, y1, color='k', linestyle=':', lw=1, alpha=1)  # 下
        plt.plot(xkn, yk, color='k', linestyle=':', lw=1, alpha=1)  # 右
        plt.plot(xk, ykn, color='k', linestyle=':', lw=1, alpha=1)  # 上

        return
    # 画虚线框
    width2 = times * width
    height2 = times * height
    MyFrame(x0, y0, width, height)
    MyFrame(x1, y1, width2, height2)

    # 连接两个虚线框
    xp = np.linspace(x0 + width, x1, num=20)
    yp = np.linspace(y0, y1 + height2, num=20)
    plt.plot(xp, yp, color='k', linestyle=':', lw=1, alpha=1)

    # 小虚框内各点坐标
    XDottedLine = []
    YDottedLine = []
    counter = 0
    while counter < len(mean_fpr):
        if mean_fpr[counter] > x0 and mean_fpr[counter] < (x0 + width) and mean_tpr[counter] > y0 and mean_tpr[counter] < (y0 + height):
            XDottedLine.append(mean_fpr[counter])
            YDottedLine.append(mean_tpr[counter])
        counter = counter + 1

    # 画虚线框内的点
    # 把小虚框内的任一点减去小虚框左下角点生成相对坐标，再乘以倍数（4）加大虚框左下角点
    counter = 0
    while counter < len(XDottedLine):
        XDottedLine[counter] = (XDottedLine[counter] - x0) * times + x1
        YDottedLine[counter] = (YDottedLine[counter] - y0) * times + y1
        counter = counter + 1


    plt.plot(XDottedLine, YDottedLine, color=color, lw=thickness, alpha=1)
    return

def MyConfusionMatrix(y_real,y_predict):
    from sklearn.metrics import confusion_matrix
    CM = confusion_matrix(y_real, y_predict)
    print(CM)
    CM = CM.tolist()
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    print('TN:%d, FP:%d, FN:%d, TP:%d' % (TN, FP, FN, TP))
    Acc = (TN + TP) / (TN + TP + FN + FP)
    Sen = TP / (TP + FN)
    Spec = TN / (TN + FP)
    Prec = TP / (TP + FP)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    # 分母可能出现0，需要讨论待续
    print('Acc:', round(Acc, 4))
    print('Sen:', round(Sen, 4))
    print('Spec:', round(Spec, 4))
    print('Prec:', round(Prec, 4))
    print('MCC:', round(MCC, 4))
    Result = []
    Result.append(round(Acc, 4))
    Result.append(round(Sen, 4))
    Result.append(round(Spec, 4))
    Result.append(round(Prec, 4))
    Result.append(round(MCC, 4))
    return Result

def MyAverage(matrix):
    SumAcc = 0
    SumSen = 0
    SumSpec = 0
    SumPrec = 0
    SumMcc = 0
    counter = 0
    while counter < len(matrix):
        SumAcc = SumAcc + matrix[counter][0]
        SumSen = SumSen + matrix[counter][1]
        SumSpec = SumSpec + matrix[counter][2]
        SumPrec = SumPrec + matrix[counter][3]
        SumMcc = SumMcc + matrix[counter][4]
        counter = counter + 1
    print('AverageAcc:',SumAcc / len(matrix))
    print('AverageSen:', SumSen / len(matrix))
    print('AverageSpec:', SumSpec / len(matrix))
    print('AveragePrec:', SumPrec / len(matrix))
    print('AverageMcc:', SumMcc / len(matrix))
    return

def MyStd(result):
    import numpy as np
    NewMatrix = []
    counter = 0
    while counter < len(result[0]):
        row = []
        NewMatrix.append(row)
        counter = counter + 1
    counter = 0
    while counter < len(result):
        counter1 = 0
        while counter1 < len(result[counter]):
            NewMatrix[counter1].append(result[counter][counter1])
            counter1 = counter1 + 1
        counter = counter + 1
    StdList = []
    MeanList = []
    counter = 0
    while counter < len(NewMatrix):
        # std
        arr_std = np.std(NewMatrix[counter], ddof=1)
        StdList.append(arr_std)
        # mean
        arr_mean = np.mean(NewMatrix[counter])
        MeanList.append(arr_mean)
        counter = counter + 1
    result.append(MeanList)
    result.append(StdList)
    # 换算成百分比制
    counter = 0
    while counter < len(result):
        counter1 = 0
        while counter1 < len(result[counter]):
            result[counter][counter1] = round(result[counter][counter1] * 100, 2)
            counter1 = counter1 + 1
        counter = counter + 1
    return result

precisions = []
average_precisions = []
mean_fpr = np.linspace(0, 1, 1000)
i = 0
colorlist = ['red', 'gold', 'purple', 'green', 'blue', 'black']

# 用于保存混淆矩阵
AllResult = []
Ps = []
Rs = []
RPs = []
mean_R = np.linspace(0, 1, 1000)
Reals = []
PredictionProbs = []

counter0 = 0
while counter0 < 5:
    # print(i)
    # 读取文件
    RealAndPrediction = []
    RealAndPredictionProb = []
    RAPName = str(counter0) + '.csv'
    RAPNameProb = str(counter0) + 'Prob.csv'
    ReadMyCsv(RealAndPrediction, RAPName)
    ReadMyCsv(RealAndPredictionProb, RAPNameProb)
    # 生成Real和Prediction
    Real = []
    Prediction = []
    PredictionProb = []
    counter = 0
    while counter < len(RealAndPrediction):
        Real.append(int(RealAndPrediction[counter][0]))
        Prediction.append(RealAndPrediction[counter][1])
        PredictionProb.append(RealAndPredictionProb[counter][1])
        counter = counter + 1

    average_precision = average_precision_score(Real, PredictionProb)
    precision, recall, _ = precision_recall_curve(Real, PredictionProb)

    Ps.append(interp(mean_R, precision, recall))
    RPs.append(average_precision)

    Reals.append(Real)
    PredictionProbs.append(PredictionProb)
    # 阶梯状
    # plt.step(recall, precision, color=colorlist[i], alpha=0.4, where='post')
    # 弧线
    plt.plot(recall, precision, lw=1.5, alpha=0.8, color=colorlist[i],
             label='fold %d (AUPR = %0.4f)' % (i, average_precision))

    print('average_precision', average_precision)

    i += 1
    counter0 = counter0 + 1



# # 画均值
# print(np.array(Ps).shape)
mean_P = np.mean(Ps, axis=0)
# print(mean_P)
mean_RPs = np.mean(RPs, axis=0)
# print(mean_RPs)
std_RPs = np.std(RPs)
plt.plot(mean_P, mean_R, color='black',
         label=r'Mean (AUPR = %0.4f)' % (mean_RPs),
         lw=2, alpha=1)
# MyEnlarge(0, 0.7, 0.25, 0.25, 0.5, 0, 2, mean_P, mean_R, 2, colorlist[5])



PAndR = []
counter = 0
while counter < len(mean_P):
    pair = []
    pair.append(mean_P[counter])
    pair.append(mean_R[counter])
    PAndR.append(pair)
    counter = counter + 1
storFile(PAndR, 'PAndRAttributeManner.csv')


plt.xlabel('Recall',fontsize=13)
plt.ylabel('Precision',fontsize=13)
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
# plt.title('2-class Precision-Recall curve: AP={0:0.4f}'.format(mean_RPs))
# 画网格
# plt.grid(linestyle='-')
# 画对角线
plt.plot([1, 0], [0, 1], color='navy', lw=2, linestyle='--')
plt.legend(bbox_to_anchor=(0.555, 0.45), fontsize='large')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])

plt.savefig('PR-5fold.svg')
plt.savefig('PR-5fold.tif')
plt.show()























