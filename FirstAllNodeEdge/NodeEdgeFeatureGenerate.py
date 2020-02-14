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

def MyNodeEdgeAttributeGenerate():
    # 数据
    CircDiseaseMergeAssociation = []
    ReadMyCsv(CircDiseaseMergeAssociation, "FirstAllNodeEdge\CircDiseaseMergeAssociation.csv")
    print('CircDiseaseMergeAssociation[0]', CircDiseaseMergeAssociation[0])

    CircMiSomamiRAssociation = []
    ReadMyCsv(CircMiSomamiRAssociation, "FirstAllNodeEdge\CircMiSomamiRAssociation.csv")
    print('CircMiSomamiRAssociation[0]', CircMiSomamiRAssociation[0])

    DiseaseMDisGeNETAssociation = []
    ReadMyCsv(DiseaseMDisGeNETAssociation, "FirstAllNodeEdge\DiseaseMDisGeNETAssociation.csv")
    print('DiseaseMDisGeNETAssociation[0]', DiseaseMDisGeNETAssociation[0])

    DiseaseMicrobeHMDADAssociation = []
    ReadMyCsv(DiseaseMicrobeHMDADAssociation, "FirstAllNodeEdge\DiseaseMicrobeHMDADAssociation.csv")
    print('DiseaseMicrobeHMDADAssociation[0]', DiseaseMicrobeHMDADAssociation[0])

    DrugDiseaseSCMFDDAssociation = []
    ReadMyCsv(DrugDiseaseSCMFDDAssociation, "FirstAllNodeEdge\DrugDiseaseSCMFDDAssociation.csv")
    print('DrugDiseaseSCMFDDAssociation[0]', DrugDiseaseSCMFDDAssociation[0])

    DrugMicrobeAssociation = []
    ReadMyCsv(DrugMicrobeAssociation, "FirstAllNodeEdge\DrugMicrobeAssociation.csv")
    print('DrugMicrobeAssociation[0]', DrugMicrobeAssociation[0])

    DrugMPharmGKBAssociation = []
    ReadMyCsv(DrugMPharmGKBAssociation, "FirstAllNodeEdge\DrugMPharmGKBAssociation.csv")
    print('DrugMPharmGKBAssociation[0]', DrugMPharmGKBAssociation[0])

    DrugProteinDrugBankAssociationThreshold5 = []
    ReadMyCsv(DrugProteinDrugBankAssociationThreshold5, "FirstAllNodeEdge\DrugProteinDrugBankAssociationThreshold5.csv")
    print('DrugProteinDrugBankAssociationThreshold5[0]', DrugProteinDrugBankAssociationThreshold5[0])

    LncDiseaseMergeAssociation = []
    ReadMyCsv(LncDiseaseMergeAssociation, "FirstAllNodeEdge\LncDiseaseMergeAssociation.csv")
    print('LncDiseaseMergeAssociation[0]', LncDiseaseMergeAssociation[0])

    LncMiSNPAssociation = []
    ReadMyCsv(LncMiSNPAssociation, "FirstAllNodeEdge\LncMiSNPAssociation.csv")
    print('LncMiSNPAssociation[0]', LncMiSNPAssociation[0])

    LncMLncRNA2TargetAssociation = []
    ReadMyCsv(LncMLncRNA2TargetAssociation, "FirstAllNodeEdge\LncMLncRNA2TargetAssociation.csv")
    print('LncMLncRNA2TargetAssociation[0]', LncMLncRNA2TargetAssociation[0])

    LncProteinNPInterAssociation = []
    ReadMyCsv(LncProteinNPInterAssociation, "FirstAllNodeEdge\LncProteinNPInterAssociation.csv")
    print('LncProteinNPInterAssociation[0]', LncProteinNPInterAssociation[0])

    MiDiseaseCuiAssociation = []
    ReadMyCsv(MiDiseaseCuiAssociation, "FirstAllNodeEdge\MiDiseaseCuiAssociation.csv")
    print('MiDiseaseCuiAssociation[0]', MiDiseaseCuiAssociation[0])

    MiDrugSM2Association = []
    ReadMyCsv(MiDrugSM2Association, "FirstAllNodeEdge\MiDrugSM2Association.csv")
    print('MiDrugSM2Association[0]', MiDrugSM2Association[0])

    MiMNMiTarbaseAssociation = []
    ReadMyCsv(MiMNMiTarbaseAssociation, "FirstAllNodeEdge\MiMNMiTarbaseAssociation.csv")
    print('MiMNMiTarbaseAssociation[0]', MiMNMiTarbaseAssociation[0])

    MiProteinMergeAssociation = []
    ReadMyCsv(MiProteinMergeAssociation, "FirstAllNodeEdge\MiProteinMergeAssociation.csv")
    print('MiProteinMergeAssociation[0]', MiProteinMergeAssociation[0])

    MProteinNCBIAssociation = []
    ReadMyCsv(MProteinNCBIAssociation, "FirstAllNodeEdge\MProteinNCBIAssociation.csv")
    print('PPI[0]', MProteinNCBIAssociation[0])

    PPI = []
    ReadMyCsv(PPI, "FirstAllNodeEdge\PPI.csv")
    print('PPI[0]', PPI[0])

    # AllEdge
    AllEdge = []
    AllEdge.extend(CircDiseaseMergeAssociation)
    AllEdge.extend(CircMiSomamiRAssociation)
    AllEdge.extend(DiseaseMDisGeNETAssociation)
    AllEdge.extend(DiseaseMicrobeHMDADAssociation)
    AllEdge.extend(DrugDiseaseSCMFDDAssociation)
    AllEdge.extend(DrugMicrobeAssociation)
    AllEdge.extend(DrugMPharmGKBAssociation)
    AllEdge.extend(DrugProteinDrugBankAssociationThreshold5)
    AllEdge.extend(LncDiseaseMergeAssociation)
    AllEdge.extend(LncMiSNPAssociation)
    AllEdge.extend(LncMLncRNA2TargetAssociation)
    AllEdge.extend(LncProteinNPInterAssociation)
    AllEdge.extend(MiDiseaseCuiAssociation)
    AllEdge.extend(MiDrugSM2Association)
    AllEdge.extend(MiMNMiTarbaseAssociation)
    AllEdge.extend(MiProteinMergeAssociation)
    AllEdge.extend(MProteinNCBIAssociation)
    AllEdge.extend(PPI)

    print('len(AllEdge)', len(AllEdge))
    print('AllEdge[0]', AllEdge[0])
    StorFile(AllEdge, 'FirstAllNodeEdge\AllEdge.csv')

    # 节点
    AllCircKmer = []
    ReadMyCsv(AllCircKmer, "FirstAllNodeEdge\AllCircKmer.csv")
    AllCirc = np.array(AllCircKmer)[:, 0]
    print('len(AllCirc)', len(AllCirc))
    print('AllCirc[0]', AllCirc[0])

    AllDiseaseFeature = []
    ReadMyCsv(AllDiseaseFeature, "FirstAllNodeEdge\AllDiseaseFeatureDAGLocal.csv")
    AllDisease = np.array(AllDiseaseFeature)[:, 0]
    print('len(AllDisease)', len(AllDisease))
    print('AllDisease[0]', AllDisease[0])

    AllDrugFeature = []
    ReadMyCsv(AllDrugFeature, "FirstAllNodeEdge\AllDrugFeature.csv")
    AllDrug = np.array(AllDrugFeature)[:, 0]
    print('len(AllDrug)', len(AllDrug))
    print('AllDrug[0]', AllDrug[0])

    AllLncKmer = []
    ReadMyCsv(AllLncKmer, "FirstAllNodeEdge\AllLncKmer.csv")
    AllLnc = np.array(AllLncKmer)[:, 0]
    print('len(AllLnc)', len(AllLnc))
    print('AllLnc[0]', AllLnc[0])

    AllMicrobeFeature = []
    ReadMyCsv(AllMicrobeFeature, "FirstAllNodeEdge\AllMicrobeFeatureDAGLocal.csv")
    AllMicrobe = np.array(AllMicrobeFeature)[:, 0]
    print('len(AllMicrobe)', len(AllMicrobe))
    print('AllMicrobe[0]', AllMicrobe[0])

    AllMiKmer = []
    ReadMyCsv(AllMiKmer, "FirstAllNodeEdge\AllMiKmer.csv")
    AllMi = np.array(AllMiKmer)[:, 0]
    print('len(AllMi)', len(AllMi))
    print('AllMi[0]', AllMi[0])

    AllMKmer = []
    ReadMyCsv(AllMKmer, "FirstAllNodeEdge\AllMKmer.csv")
    AllM = np.array(AllMKmer)[:, 0]
    print('len(AllM)', len(AllM))
    print('AllM[0]', AllM[0])

    AllProteinKmer = []
    ReadMyCsv(AllProteinKmer, "FirstAllNodeEdge\AllProteinKmer.csv")
    AllProtein = np.array(AllProteinKmer)[:, 0]
    print('len(AllProtein)', len(AllProtein))
    print('AllProtein[0]', AllProtein[0])

    # AllNode
    AllNode = []
    AllNode.extend(AllCirc)
    AllNode.extend(AllDisease)
    AllNode.extend(AllDrug)
    AllNode.extend(AllLnc)
    AllNode.extend(AllMicrobe)
    AllNode.extend(AllMi)
    AllNode.extend(AllM)
    AllNode.extend(AllProtein)
    print('len(AllNode)', len(AllNode))
    counter = 0
    while counter < len(AllNode):
        pair = []
        pair.append(AllNode[counter])
        AllNode[counter] = pair
        counter = counter + 1
    print(AllNode[0])
    StorFile(AllNode, 'FirstAllNodeEdge\AllNode.csv')

    # AllNodeAttribute
    AllNodeAttribute = []
    AllNodeAttribute.extend(AllCircKmer)
    AllNodeAttribute.extend(AllDiseaseFeature)
    AllNodeAttribute.extend(AllDrugFeature)
    AllNodeAttribute.extend(AllLncKmer)
    AllNodeAttribute.extend(AllMicrobeFeature)
    AllNodeAttribute.extend(AllMiKmer)
    AllNodeAttribute.extend(AllMKmer)
    AllNodeAttribute.extend(AllProteinKmer)

    StorFile(AllNodeAttribute, 'FirstAllNodeEdge\AllNodeAttribute.csv')
    print(np.array(AllNodeAttribute).shape)

    return AllNode, AllEdge, AllNodeAttribute
