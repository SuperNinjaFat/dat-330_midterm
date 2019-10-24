import csv
import math
import time
import glob
import random
import operator
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def loadDatasets():
    le = LabelEncoder()
    warnings.filterwarnings('ignore')
    diseases = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'COADREAD',
                'DLBC', 'ESCA', 'FPPP', 'GBM', 'GBMLGG', 'HNSC', 'KICH',
                'KIPAN', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC',
                'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM',
                'STAD', 'STES', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']
    patient_info = {}
    #Testing just ACC files, diseases.keys() for full read
    for disease in diseases:
        #Collect all clinical data
        filename = 'transposed data/' + disease + '.clin.merged(flipped).csv'
        file = pd.read_csv(filename)
        for i in range(len(file)):
            patient_info[file['patient.bcr_patient_barcode'][i]] = [
                file['admin.disease_code'][i], file['patient.gender'][i],
                round(file['patient.days_to_birth'][i]/-365, 2)]
        #Attach mutation data to matching clinical data
        for filename in glob.glob('mutations/gdac.broadinstitute.org_' + disease + '.Mutation_Packager_Calls.Level_3.2016012800.0.0/*.maf.txt'):
            file = pd.read_csv(filename, sep='\\t')
            try:
                patient_info[filename.split('\\')[-1].split('.')[0][:-3].lower()][3] += (len(file))
            except IndexError:
                patient_info[filename.split('\\')[-1].split('.')[0][:-3].lower()].append(len(file))
            except KeyError:
                continue
        print(disease)
    patient_info = {k:v for k,v in patient_info.items() if len(v) > 3}
    #Transfer patient data to dataframe
    patient_data = pd.DataFrame(data=patient_info)
    patient_data = patient_data.transpose()
    patient_data = patient_data.rename(columns={0: 'Disease', 1: 'Gender', 2: 'Age', 3:'Mutations'})

    #Weigh mutation data
    avg_mut = patient_data['Mutations'].mean()
    for x in range(len(patient_data['Mutations'])):
        if patient_data['Mutations'][x] > avg_mut:
            patient_data['Mutations'][x] = 1
        else:
            patient_data['Mutations'][x] = 0

    #Remove any rows with blank data
    patient_data = patient_data.dropna()

    #Encode non-int columns
    for x in range(len(patient_data.columns)):
        if x not in [2, 4]:
            patient_data.iloc[:,x] = le.fit_transform(patient_data.iloc[:,x])

    y = patient_data.iloc[:,-1].astype('int')
    X = patient_data.drop(['Mutations'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    return X_test, y_test, X_train, y_train

def k_neighbors(X_test, y_test, X_train, y_train, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_knn = knn.predict(X_test)

    correct = 0
    for i in range(len(y_knn)):
        if y_test.iloc[i] == y_knn[i]:
            correct += 1
    accuracy = (correct / float(len(y_test))) * 100.0
    print('KNN Accuracy: ' + repr(round(accuracy, 2)) + '%')

def decision_tree(X_test, y_test, X_train, y_train, k):
    dt = DecisionTreeClassifier(max_depth=k, random_state=1)
    dt.fit(X_train, y_train)
    y_dt = dt.predict(X_test)

    correct = 0
    for i in range(len(y_dt)):
        if y_test.iloc[i] == y_dt[i]:
            correct += 1
    accuracy = (correct / float(len(y_test))) * 100.0
    print('Decision Tree Accuracy: ' + repr(round(accuracy, 2)) + '%')


if __name__ == "__main__":
    start = time.time()
    X_test, y_test, X_train, y_train = loadDatasets()

    # KNN
    k_neighbors(X_test, y_test, X_train, y_train, 3)

    # Decision trees
    decision_tree(X_test, y_test, X_train, y_train, 6)
    print("(Time to complete: " + str(round(time.time() - start, 1)) + "s)")



    
