import src.utils.plot_utils as plot_utils
import file_read

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

feature_names = file_read.df_train_ori.columns
feature_names = feature_names.tolist()
del feature_names[28] # delete 'RUL'

# plot_utils.plot_correlation_map(file_read.df_train_clip, 'train clipped dataset feature')
# plot_utils.plot_correlation_map(file_read.df_test_clip, 'test clipped dataset feature')

# plot_utils.plot_feature_distri(file_read.df_train_ori, file_read.df_test_ori, 'original data feature')
# plot_utils.plot_feature_distri(file_read.df_train_clip, file_read.df_test_clip, 'clip data feature')

# fig, axes = plt.subplots(nrows=7, ncols=3, figsize=(18, 28))
# for i in range(1, 22):
#     try:
#         row = (i - 1) // 3
#         col = (i - 1) % 3
#         plot_utils.plot_signal(file_read.df_train_raw_RUL, plot_utils.Sensor_dictionary, 's_' + str(i), axes[row, col])
#     except:
#         pass

# plt.subplots_adjust(hspace=0.7)
# plt.show()

'''
feature importance
'''
X = file_read.Train_X
y = file_read.Train_y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)

'''
random forest feature importance
'''
# plt.barh(feature_names, rf.feature_importances_)
# plt.title('Random Forest Feature Importance')

'''
permutation feature importance
'''
perm_importance = permutation_importance(rf, X_test, y_test)
plt.barh(feature_names, perm_importance.importances_mean)
plt.title('Permutation Feature Importance')

plt.show()