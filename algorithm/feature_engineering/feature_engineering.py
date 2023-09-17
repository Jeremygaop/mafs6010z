from helper.helper import *
from utils.constant import ROOT_PATH, DATAPath, PICKLE_PATH, DATA_FOR_MODELLING_PATH



if __name__ == '__main__':
    # create folders if PICKLE_PATH does not exist
    if not os.path.exists(PICKLE_PATH):
        os.makedirs(PICKLE_PATH)
    # preprocess data and save to pickles
    preprocess_bureau_balance_and_bureau(dump_to_pickle=True).main()
    preprocess_application_train_test(dump_to_pickle=True).main()
    preprocess_installments_payments(dump_to_pickle=True).main()
    preprocess_previous_application(dump_to_pickle=True).main()
