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

def loadDatasets():
    warnings.filterwarnings('ignore')
    diseases = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'COADREAD',
                'DLBC', 'ESCA', 'FPPP', 'GBM', 'GBMLGG', 'HNSC', 'KICH',
                'KIPAN', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC',
                'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM',
                'STAD', 'STES', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']
    patient_info = {}
    #Testing just ACC files, diseases.keys() for full read
    for disease in diseases[:3]:
        #Collect all clinical data
        filename = 'transposed data/' + disease + '.clin.merged(flipped).csv'
        file = pd.read_csv(filename)
        for i in range(len(file)):
            patient_info[file['patient.bcr_patient_barcode'][i]] = [
                file['admin.disease_code'][i], file['patient.gender'][i],
                -file['patient.days_to_birth'][i], file['patient.stage_event.pathologic_stage'][i]]
        #Attach mutation data to matching clinical data
        for filename in glob.glob('mutations/gdac.broadinstitute.org_' + disease + '.Mutation_Packager_Calls.Level_3.2016012800.0.0/*.maf.txt'):
            file = pd.read_csv(filename, sep='\\t')
            try:
                patient_info[filename.split('\\')[-1].split('.')[0][:-3].lower()][4] += (len(file))
            except IndexError:
                patient_info[filename.split('\\')[-1].split('.')[0][:-3].lower()].append(len(file))
    patient_info = {k:v for k,v in patient_info.items() if len(v) > 4}
    #Transfer patient data to dataframe
    patient_data = pd.DataFrame(data=patient_info)
    patient_data = patient_data.transpose()
    patient_data = patient_data.rename(columns={0: 'Disease', 1: 'Gender', 2:'Age', 3:'Stage', 4:'Mutations'})

    return patient_data

    #Important data
    ##print(clinical_data['ACC']['patient.days_to_death']) #float
    ##print(clinical_data['ACC']['patient.radiation_therapy']) #boolean
    ##print(clinical_data['ACC']['patient.ethnicity']) #Encode
    ##print(clinical_data['ACC']['patient.primary_pathology.residual_tumor']) #Encode
    ##print(clinical_data['ACC']['patient.primary_pathology.lymph_node_examined_count']) #float
    ##print(clinical_data['ACC']['patient.primary_pathology.histological_type']) #Encode

if __name__ == "__main__":
    #If you are... (gender, ethnicity, age, etc...)... you are more/less likely to have
    #more mutations, mutations for... (gene types)... genes.
    ###Percentage for mutation overall, and for specific genes
    start = time.time()
    patient_data = loadDatasets()
    print(patient_data)
    print("(Time to complete: " + str(round(time.time() - start, 1)) + "s)")



    
