import src.utils.file_utils as file_utils
from torch.utils.data import TensorDataset, DataLoader

method_data_preprocess = 'WINDOW'
WINDOW_SIZE = 40
RUL_clip_value = 105
BATCH_SIZE = 512

Num_Slurm_HPO = 1101965
method_Regression = 'DSPP'
model_state_file_name = f'FD001_model_state_{Num_Slurm_HPO}_{method_Regression}.pth'
model_state_path = f'/home/linyuhan/Dokumente/Masterarbeit/dataset/CMAPSS/{model_state_file_name}'

TRAIN_DATASET_NAME = 'train_FD001'
TRAIN_FILE_PATH = '/home/linyuhan/Dokumente/Masterarbeit/dataset/CMAPSS/' + TRAIN_DATASET_NAME + '.txt'
TEST_DATASET_NAME = 'test_FD001'
RUL_DATASET_NAME = 'RUL_FD001'
TEST_FILE_PATH = '/home/linyuhan/Dokumente/Masterarbeit/dataset/CMAPSS/' + TEST_DATASET_NAME + '.txt'
RUL_FILE_PATH = '/home/linyuhan/Dokumente/Masterarbeit/dataset/CMAPSS/' + RUL_DATASET_NAME + '.txt'

# model_state_file_name = f'FD002_model_state_{1091182}_v2.pth'
# model_state_path = f'/home/linyuhan/Dokumente/Masterarbeit/dataset/CMAPSS/{model_state_file_name}'

# TRAIN_DATASET_NAME = 'train_FD002'
# TRAIN_FILE_PATH = '/home/linyuhan/Dokumente/Masterarbeit/dataset/CMAPSS/' + TRAIN_DATASET_NAME + '.txt'
# TEST_DATASET_NAME = 'test_FD002'
# RUL_DATASET_NAME = 'RUL_FD002'
# TEST_FILE_PATH = '/home/linyuhan/Dokumente/Masterarbeit/dataset/CMAPSS/' + TEST_DATASET_NAME + '.txt'
# RUL_FILE_PATH = '/home/linyuhan/Dokumente/Masterarbeit/dataset/CMAPSS/' + RUL_DATASET_NAME + '.txt'

'''
Build Dataset for WINDOW method
'''
if method_data_preprocess == 'WINDOW':
    ### train dataset ###
    df_train_raw = file_utils.read_original_file(TRAIN_FILE_PATH)
    df_train_raw_RUL = file_utils.add_RUL_column(df_train_raw)
    df_train_ori = file_utils.drop_col(df_train_raw_RUL, method_data_preprocess)
    df_train_clip = file_utils.clip_row(df_train_ori, RUL_clip_value)
    df_train = file_utils.add_history_data(df_train_clip, WINDOW_SIZE)

    X_train, y_train = file_utils.drop_not_split(df_train) #float64

    # data scale
    X_train_s = file_utils.data_scale(X_train) #float64
    # build train/val dataloader
    Train_X = file_utils.reshape_data(X_train_s)
    Train_y = file_utils.reshape_data(y_train)

    train_dataset = TensorDataset(Train_X, Train_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # df_train.to_csv('df_train.csv', index=False)

    ### test dataset ###
    df_test_raw = file_utils.read_original_file(TEST_FILE_PATH)
    df_RUL = file_utils.read_RUL_file(RUL_FILE_PATH)
    df_test_raw_RUL = file_utils.add_RUL_column_test(df_test_raw, df_RUL)
    df_test_ori = file_utils.drop_col(df_test_raw_RUL, method_data_preprocess)
    df_test_clip = file_utils.clip_row(df_test_ori, RUL_clip_value)
    df_test = file_utils.add_history_data(df_test_clip, WINDOW_SIZE)

    X_test, y_test= file_utils.drop_not_split(df_test)

    # data scale
    X_test_s = file_utils.data_scale(X_test)
    # build train/val dataloader
    Test_X = file_utils.reshape_data(X_test_s)
    Test_y = file_utils.reshape_data(y_test)

    test_dataset = TensorDataset(Test_X, Test_y)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

'''
Build Dataset for Rolling Mean method
'''
if method_data_preprocess == 'ROLLING':
    ### train dataset ###
    df_train_raw = file_utils.read_original_file(TRAIN_FILE_PATH)
    df_train_mean = file_utils.add_rolling_mean(df_train_raw)
    df_train_raw_RUL = file_utils.add_RUL_column(df_train_mean)
    df_train_ori = file_utils.drop_col(df_train_raw_RUL, method_data_preprocess)
    df_train_clip = file_utils.clip_row(df_train_ori, RUL_clip_value)
    df_train = df_train_clip

    X_train, y_train = file_utils.drop_not_split(df_train)

    # data scale
    X_train_s = file_utils.data_scale(X_train)
    # build train/val dataloader
    Train_X = file_utils.reshape_data(X_train_s)
    Train_y = file_utils.reshape_data(y_train)

    train_dataset = TensorDataset(Train_X, Train_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # df_train.to_csv('df_train.csv', index=False)

    ### test dataset ###
    df_test_raw = file_utils.read_original_file(TEST_FILE_PATH)
    df_test_mean = file_utils.add_rolling_mean(df_test_raw)
    df_RUL = file_utils.read_RUL_file(RUL_FILE_PATH)
    df_test_raw_RUL = file_utils.add_RUL_column_test(df_test_mean, df_RUL)
    df_test_ori = file_utils.drop_col(df_test_raw_RUL, method_data_preprocess)
    df_test_clip = file_utils.clip_row(df_test_ori, RUL_clip_value)
    df_test = df_test_clip

    X_test, y_test= file_utils.drop_not_split(df_test)

    # data scale
    X_test_s = file_utils.data_scale(X_test)
    # build train/val dataloader
    Test_X = file_utils.reshape_data(X_test_s)
    Test_y = file_utils.reshape_data(y_test)

    test_dataset = TensorDataset(Test_X, Test_y)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)