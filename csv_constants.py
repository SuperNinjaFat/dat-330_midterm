import os
import requests
import tarfile

HTTP_PREFIX = "http://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/"
HTTP_PREPREFIX = "/20160128/"
FILE_PREFIX = "gdac.broadinstitute.org_"
FILE_POSTFIX = ".Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz"
DISEASES = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'COADREAD',
                'DLBC', 'ESCA', 'FPPP', 'GBM', 'GBMLGG', 'HNSC', 'KICH',
                'KIPAN', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC',
                'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM',
                'STAD', 'STES', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']

DIR_BASE = os.path.dirname(__file__)


def download():
    for name in DISEASES:
        filename = (FILE_PREFIX + name + FILE_POSTFIX)
        downloadTar(filename, (HTTP_PREFIX + name + HTTP_PREPREFIX + filename))
        print(trytar(filename))


def trytar(filename):
    try:
        tf = tarfile.open(os.path.join(os.path.join(DIR_BASE, 'raw_tar'), filename))
    except tarfile.ReadError as e:
        return e
    tf.extractall(os.path.join(DIR_BASE, 'mutations'))
    return filename


def downloadTar(output, address):
    f = open(os.path.join(os.path.join(DIR_BASE, 'raw_tar'), output), 'wb')
    f.write(requests.get(address).content)
    f.close()

download()