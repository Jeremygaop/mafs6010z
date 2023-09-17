from datetime import datetime
import pickle
from utils.data_process_helper import reduce_mem_usage
from utils.constant import ROOT_PATH, DATAPath, PICKLE_PATH, DATA_FOR_MODELLING_PATH
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')


def merge_all_tables(application_train, application_test, bureau_aggregated, previous_aggregated,
                     installments_aggregated, pos_aggregated, cc_aggregated):

    # merging application_train and application_test with Aggregated bureau table
    app_train_merged = application_train.merge(bureau_aggregated, on='SK_ID_CURR', how='left')
    app_test_merged = application_test.merge(bureau_aggregated, on='SK_ID_CURR', how='left')
    # merging with aggregated previous_applications
    app_train_merged = app_train_merged.merge(previous_aggregated, on='SK_ID_CURR', how='left')
    app_test_merged = app_test_merged.merge(previous_aggregated, on='SK_ID_CURR', how='left')
    # merging with aggregated installments tables
    app_train_merged = app_train_merged.merge(installments_aggregated, on='SK_ID_CURR', how='left')
    app_test_merged = app_test_merged.merge(installments_aggregated, on='SK_ID_CURR', how='left')
    # merging with aggregated POS_Cash balance table
    app_train_merged = app_train_merged.merge(pos_aggregated, on='SK_ID_CURR', how='left')
    app_test_merged = app_test_merged.merge(pos_aggregated, on='SK_ID_CURR', how='left')
    # merging with aggregated credit card table
    app_train_merged = app_train_merged.merge(cc_aggregated, on='SK_ID_CURR', how='left')
    app_test_merged = app_test_merged.merge(cc_aggregated, on='SK_ID_CURR', how='left')

    return reduce_mem_usage(app_train_merged), reduce_mem_usage(app_test_merged)


def create_new_features(data):

    # previous applications columns
    prev_annuity_columns = [ele for ele in previous_aggregated.columns if 'AMT_ANNUITY' in ele]
    for col in prev_annuity_columns:
        data['PREV_' + col + '_INCOME_RATIO'] = data[col] / (data['AMT_INCOME_TOTAL'] + 0.00001)
    prev_goods_columns = [ele for ele in previous_aggregated.columns if 'AMT_GOODS' in ele]
    for col in prev_goods_columns:
        data['PREV_' + col + '_INCOME_RATIO'] = data[col] / (data['AMT_INCOME_TOTAL'] + 0.00001)

    # credit_card_balance columns
    cc_amt_principal_cols = [ele for ele in cc_aggregated.columns if 'AMT_RECEIVABLE_PRINCIPAL' in ele]
    for col in cc_amt_principal_cols:
        data['CC_' + col + '_INCOME_RATIO'] = data[col] / (data['AMT_INCOME_TOTAL'] + 0.00001)
    cc_amt_recivable_cols = [ele for ele in cc_aggregated.columns if 'AMT_RECIVABLE' in ele]
    for col in cc_amt_recivable_cols:
        data['CC_' + col + '_INCOME_RATIO'] = data[col] / (data['AMT_INCOME_TOTAL'] + 0.00001)
    cc_amt_total_receivable_cols = [ele for ele in cc_aggregated.columns if 'TOTAL_RECEIVABLE' in ele]
    for col in cc_amt_total_receivable_cols:
        data['CC_' + col + '_INCOME_RATIO'] = data[col] / (data['AMT_INCOME_TOTAL'] + 0.00001)

    # installments_payments columns
    installments_payment_cols = [ele for ele in installments_aggregated.columns if
                                 'AMT_PAYMENT' in ele and 'RATIO' not in ele and 'DIFF' not in ele]
    for col in installments_payment_cols:
        data['INSTALLMENTS_' + col + '_INCOME_RATIO'] = data[col] / (data['AMT_INCOME_TOTAL'] + 0.00001)
    # https://www.kaggle.com/c/home-credit-default-risk/discussion/64821
    installments_max_installment = ['AMT_INSTALMENT_MEAN_MAX', 'AMT_INSTALMENT_SUM_MAX']
    for col in installments_max_installment:
        data['INSTALLMENTS_ANNUITY_' + col + '_RATIO'] = data['AMT_ANNUITY'] / (data[col] + 0.00001)

    # POS_CASH_balance features have been created in its own dataframe itself

    # bureau and bureau_balance columns
    bureau_days_credit_cols = [ele for ele in bureau_aggregated.columns if
                               'DAYS_CREDIT' in ele and 'ENDDATE' not in ele and 'UPDATE' not in ele]
    for col in bureau_days_credit_cols:
        data['BUREAU_' + col + '_EMPLOYED_DIFF'] = data[col] - data['DAYS_EMPLOYED']
        data['BUREAU_' + col + '_REGISTRATION_DIFF'] = data[col] - data['DAYS_REGISTRATION']
    bureau_overdue_cols = [ele for ele in bureau_aggregated.columns if 'AMT_CREDIT' in ele and 'OVERDUE' in ele]
    for col in bureau_overdue_cols:
        data['BUREAU_' + col + '_INCOME_RATIO'] = data[col] / (data['AMT_INCOME_TOTAL'] + 0.00001)
    bureau_amt_annuity_cols = [ele for ele in bureau_aggregated.columns if 'AMT_ANNUITY' in ele and 'CREDIT' not in ele]
    for col in bureau_amt_annuity_cols:
        data['BUREAU_' + col + '_INCOME_RATIO'] = data[col] / (data['AMT_INCOME_TOTAL'] + 0.00001)


def final_pickle_dump(train_data, test_data, train_file_name, test_file_name, file_directory='', verbose=True):
    if not os.path.exists(DATA_FOR_MODELLING_PATH):
        os.makedirs(DATA_FOR_MODELLING_PATH)
    if verbose:
        print("Dumping the final preprocessed data to pickle files.")
        start = datetime.now()
    with open(os.path.join(PICKLE_PATH, train_file_name + '.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(PICKLE_PATH, test_file_name + '.pkl'), 'wb') as f:
        pickle.dump(test_data, f)

    if verbose:
        print("Done.")
        print(f"Time elapsed = {datetime.now() - start}")


if __name__ == '__main__':
    # reading pickled files as dataframes
    application_train = pd.read_pickle(os.path.join(PICKLE_PATH, 'application_train_preprocessed.pkl'))
    application_test = pd.read_pickle(os.path.join(PICKLE_PATH, 'application_test_preprocessed.pkl'))
    bureau_aggregated = pd.read_pickle(os.path.join(PICKLE_PATH, 'bureau_merged_preprocessed.pkl'))
    previous_aggregated = pd.read_pickle(os.path.join(PICKLE_PATH, 'previous_application_preprocessed.pkl'))
    installments_aggregated = pd.read_pickle(os.path.join(PICKLE_PATH, 'installments_payments_preprocessed.pkl'))
    pos_aggregated = pd.read_pickle(os.path.join(PICKLE_PATH, 'POS_CASH_balance_preprocessed.pkl'))
    cc_aggregated = pd.read_pickle(os.path.join(PICKLE_PATH, 'credit_card_balance_preprocessed.pkl'))

    # merging all the tables together
    train_data, test_data = merge_all_tables(application_train, application_test, bureau_aggregated,
                                             previous_aggregated, installments_aggregated, pos_aggregated,
                                             cc_aggregated)

    create_new_features(train_data)
    create_new_features(test_data)

    print("After Pre-processing, aggregation, merging and Feature Engineering,")
    print(f"Final Shape of Training Data = {train_data.shape}")
    print(f"Final Shape of Test Data = {test_data.shape}")
    final_pickle_dump(train_data, test_data, 'train_data_final', 'test_data_final')
