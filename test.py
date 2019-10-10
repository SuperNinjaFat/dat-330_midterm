import csv
import math
import time
import random
import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier

if __name__ == "__main__":
    diseases = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'COADREAD',
                'DLBC', 'ESCA', 'FPPP', 'GBM', 'GBMLGG', 'HNSC', 'KICH',
                'KIPAN', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC',
                'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM',
                'STAD', 'STES', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']
    for disease in diseases:
        filename = disease + '.clin.merged.csv'
        with open(filename) as csvfile:
            patient_data = pd.DataFrame(columns=['age','DOD','gender','dc','rt'])
            readCSV = csv.reader(csvfile, delimiter=',')
            age = []
            DOD = []
            gender = []
            dc = []
            rt = []
            for row in readCSV:
                if repr(row[0]).split('\\')[0] == '\'patient.days_to_birth':
                    for i in repr(row[0]).replace('\'', '').replace('t', '').split('\\')[1:]:
                        try:
                            age.append(float(i)/-365)
                        except ValueError:
                            age.append('N/A')
                elif repr(row[0]).split('\\')[0] == '\'patient.days_to_death':
                    for i in repr(row[0]).replace('\'', '').replace('t', '').split('\\')[1:]:
                        try:
                            DOD.append(int(i)/365)
                        except ValueError:
                            DOD.append('N/A')
                elif repr(row[0]).split('\\')[0] == '\'patient.gender':
                    for i in repr(row[0]).replace('\'', '').replace('t', '').split('\\')[1:]:
                        gender.append(i)
                elif repr(row[0]).split('\\')[0] == '\'admin.disease_code':
                    for i in repr(row[0]).replace('\'', '').replace('t', '').split('\\')[1:]:
                        dc.append(i)
                elif repr(row[0]).split('\\')[0] == '\'patient.radiation_therapy':
                    for i in repr(row[0]).replace('\'', '').replace('t', '').split('\\')[1:]:
                        rt.append(i)
            patient_data['age'] = age
            patient_data['DOD'] = DOD
            patient_data['gender'] = gender
            patient_data['dc'] = dc
            #patient_data['rt'] = rt
            print(patient_data)

    print('total # of rows: ' + str(14729))
