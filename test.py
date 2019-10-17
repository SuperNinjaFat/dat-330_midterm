import csv
import math
import time
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

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    diseases = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'COADREAD',
                'DLBC', 'ESCA', 'FPPP', 'GBM', 'GBMLGG', 'HNSC', 'KICH',
                'KIPAN', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC',
                'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM',
                'STAD', 'STES', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']
    patient_data = {}
    for disease in diseases:
        filename = 'transposed data/' + disease + '.clin.merged(flipped).csv'
        patient_data[disease] = pd.read_csv(filename)
                
    print(patient_data)
