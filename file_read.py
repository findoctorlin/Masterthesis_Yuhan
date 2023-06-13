import src.utils.file_utils as file_utils
from torch.utils.data import TensorDataset, DataLoader

DATASET_NAME = 'train_FD001'
FILE_PATH = '/home/linyuhan/Dokumente/Masterarbeit/dataset/CMAPSS/' + DATASET_NAME + '.txt'
WINDOW_SIZE = 1

df_train_FD001 = file_utils.read_original_file(FILE_PATH)
df_train_FD001 = file_utils.add_RUL_column(df_train_FD001)
df_train_FD001 = file_utils.drop_col(df_train_FD001)
df_train_FD001 = file_utils.add_history_data(df_train_FD001, WINDOW_SIZE)
X_train, X_val, y_train, y_val = file_utils.drop_n_split(df_train_FD001, 0.1)
# data scale
X_train_s = file_utils.data_scale(X_train)
X_val_s = file_utils.data_scale(X_val)
# build dataloader
Train_X = file_utils.reshape_data(X_train_s)
Val_X = file_utils.reshape_data(X_val_s)
Train_y = file_utils.reshape_data(y_train)
Val_y = file_utils.reshape_data(y_val)

train_dataset = TensorDataset(Train_X, Train_y)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)

val_dataset = TensorDataset(Val_X, Val_y)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

# df_train_FD001.to_csv('df_train_FD001.csv', index=False)