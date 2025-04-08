import utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# loading data
df = utils.loading_data()
print(df.head())

print(df.columns)

# checking nulls and duplicates
nulls, duplicates = utils.checking_nulls_duplicates(df)
print('nulls:', nulls)
print('duplicates:', duplicates)

# splitting the features and target columns
x, y = utils.splitting_y_x(df)

# checking possible classes
utils.checking_possible_classes(y)

# converting classes into 0,1
y = utils.converting_classes_into_0_1(y)

# splitting data into train and test
x_train, x_test, y_train, y_test = utils.split_data_into_train_test(x, y)

# scaling data
scaler = MinMaxScaler()
x_train = utils.fitting_scaler(scaler, x_train)
x_test = utils.transform_using_scaler(scaler, x_test)

# fitting the model
svc_model = SVC(kernel='poly')
svc_model.fit(x_train, y_train)

# evaluating the model
y_pred = utils.model_prediction(svc_model, x_test)
print('model evaluation: \n', classification_report(y_test, y_pred))

# saving model
utils.saving_model(svc_model, scaler)
