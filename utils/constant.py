import os
# file path

# url
# get root path
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATAPath = os.path.join(ROOT_PATH, 'data')
PICKLE_PATH = os.path.join(DATAPath, 'pickles')
DATA_FOR_MODELLING_PATH = os.path.join(DATAPath, 'data_for_modelling')
