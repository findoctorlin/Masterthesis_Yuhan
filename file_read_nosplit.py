import src.utils.file_utils as file_utils
from torch.utils.data import TensorDataset, DataLoader

DATASET_NAME = 'train_FD001'
FILE_PATH = '/home/linyuhan/Dokumente/Masterarbeit/dataset/CMAPSS/' + DATASET_NAME + '.txt'
WINDOW_SIZE = 20
RUL_clip_value = 105
model_state_file_name = f'model_state_{WINDOW_SIZE}_{RUL_clip_value}.pth'
model_state_path = f'/home/linyuhan/Dokumente/Masterarbeit/dataset/CMAPSS/{model_state_file_name}'

df_train_FD001 = file_utils.read_original_file(FILE_PATH)
df_train_FD001 = file_utils.add_RUL_column(df_train_FD001)
df_train_FD001 = file_utils.drop_col(df_train_FD001)
df_train_FD001 = file_utils.clip_row(df_train_FD001, RUL_clip_value)

df_train_FD001 = file_utils.add_history_data(df_train_FD001, WINDOW_SIZE)

X_train, y_train= file_utils.drop_not_split(df_train_FD001)
# data scale
X_train_s = file_utils.data_scale(X_train)
# build dataloader
Train_X = file_utils.reshape_data(X_train_s)
Train_y = file_utils.reshape_data(y_train)

train_dataset = TensorDataset(Train_X, Train_y)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)

# df_train_FD001.to_csv('df_train_FD001.csv', index=False)