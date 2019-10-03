import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import model_selection

# Function
def countPermission(dfData, normal, signature, dangerous, miscellaneous):
    dataResult = pd.DataFrame(columns=['APK', 'Normal', 'Signature', 'Dangerous', 'Miscellaneous'])
    detected = dfData.fillna('')
    zeroDataframe = pd.DataFrame({0: 0}, index=[0])
    for index, row in detected.iterrows():
        tempData = pd.DataFrame()
        dfAPKPermission = pd.DataFrame(row[2].split(sep=','))
        dfAPKPermission.columns = ['name']
        # APK Name
        tempData = tempData.append(row.to_frame().transpose().APK.reset_index(drop=True))
        # Normal
        normal_counter = normal.name.isin(dfAPKPermission.name).to_frame().name.value_counts().to_frame()
        normal_counter = normal_counter.drop([False]).reset_index(drop=True)
        normal_counter.columns = [0]
        if normal_counter.empty:
            normal_counter = zeroDataframe

        # Signature
        signature_counter = signature.name.isin(
            dfAPKPermission.name).to_frame().name.value_counts().to_frame()
        signature_counter = signature_counter.drop([False]).reset_index(drop=True)
        signature_counter.columns = [0]
        if signature_counter.empty:
            signature_counter = zeroDataframe

        # Dangerous
        dangerous_counter = dangerous.name.isin(
            dfAPKPermission.name).to_frame().name.value_counts().to_frame()
        dangerous_counter = dangerous_counter.drop([False]).reset_index(drop=True)
        dangerous_counter.columns = [0]
        if dangerous_counter.empty:
            dangerous_counter = zeroDataframe

        # Miscellaneous
        miscellaneous_counter = miscellaneous.name.isin(
            dfAPKPermission.name).to_frame().name.value_counts().to_frame()
        miscellaneous_counter = miscellaneous_counter.drop([False]).reset_index(drop=True)
        miscellaneous_counter.columns = [0]
        if miscellaneous_counter.empty:
            miscellaneous_counter = zeroDataframe

        tempData = tempData.append(normal_counter). \
            append(signature_counter). \
            append(dangerous_counter). \
            append(miscellaneous_counter).transpose()
        tempData.columns = ['APK', 'Normal', 'Signature', 'Dangerous', 'Miscellaneous']
        dataResult = dataResult.append(tempData)
        # print(dfAPKPermission)
        print(index+1)
        # print(row[2])

    return dataResult


def getTrainAndTestDataset(dfData, dfPermission):
    addNineRow = pd.DataFrame([[0], [0], [0], [0], [0], [0], [0], [0], [0]])
    addNineRow.columns = ['status']
    counter = 0
    data = dfData['Permission']
    data = data.fillna('')
    dataResult = pd.DataFrame()

    for row in data:
        dfAPKPermission = row.split(sep=',')
        dfAPKPermission = pd.DataFrame(dfAPKPermission)
        dfAPKPermission.columns = ['name']
        dfAPKPermission = dfPermission.name.isin(dfAPKPermission.name).astype(int).to_frame()
        dfAPKPermission.columns = ['status']
        dfAPKPermission = dfAPKPermission.append(addNineRow).reset_index(drop=True)
        dfAPKPermission = dfAPKPermission.transpose()
        dataResult = dataResult.append(dfAPKPermission)
        print(counter + 1)
        counter += 1

    dataResult = dataResult.replace(1, 255).reset_index(drop=True)
    labelResult = dfData['status']
    labelResult = labelResult.to_frame()
    labelResult.columns = ['status']
    # return (dataResult, labelResult)
    return dataResult


#Android General Permission from Android Doc
android_manifest_permission = pd.read_csv('../manifestPermissionAndroid.csv')
android_manifest_permission['class'] = 'Miscellaneous'
android_groups_permission = pd.read_csv('../permissionGeneral.csv')
android_general_permission = android_groups_permission.append(android_manifest_permission).drop_duplicates(subset='name').reset_index(drop=True)
normal_permission = android_general_permission.loc[android_general_permission['class'] == 'Normal']
signature_permission = android_general_permission.loc[android_general_permission['class'] == 'Signature']
dangerous_permission = android_general_permission.loc[android_general_permission['class'] == 'Dangerous']
miscellaneous_permission = android_general_permission.loc[android_general_permission['class'] == 'Miscellaneous']



#Koodous Detected Dataset
detected = pd.read_csv('../New Clear Dataset/DetectedFixClear.csv')
# detected = detected.iloc[:19945]
detected = detected.drop(detected.columns[[3,4]],axis=1)
detected['Permission'] = detected['Permission'].str.replace(" ","")
detected['Class'] = 'detected'

#Undetected
undetected = pd.read_csv('../New Clear Dataset/UndetectedFixClear.csv')
# undetected = undetected.drop(undetected.columns[[0,2]],axis=1)
undetected['Permission'] = undetected['Permission'].str.replace(" ","")
undetected['Class'] = 'undetected'

# detectedData = countPermission(
#     detected,
#     normal_permission,
#     signature_permission,
#     dangerous_permission,
#     miscellaneous_permission)
#
# detectedData.to_csv('data_csv/Clear/PermissionCount_detected.csv',index=False)
#
# undetectedData = countPermission(
#     undetected,
#     normal_permission,
#     signature_permission,
#     dangerous_permission,
#     miscellaneous_permission)
#
# undetectedData.to_csv('data_csv/Clear/PermissionCount_undetected.csv',index=False)

detectedData = pd.read_csv('data_csv/Clear/PermissionCount_detected.csv')
undetectedData = pd.read_csv('data_csv/Clear/PermissionCount_undetected.csv')
newDetectedData = pd.read_csv('data_csv/Clear/detectedData_new.csv')
detectedKMeansResult = pd.read_csv('data_csv/Clear/kmeans_result.csv')
detectedKMeansResult.columns = ['status']

# permissionCount = detectedData.append(undetectedData,ignore_index=True)

# dataToKMeans = permissionCount[['Normal', 'Signature', 'Dangerous', 'Miscellaneous']]
# dataToKMeans = detectedData[['Normal', 'Signature', 'Dangerous', 'Miscellaneous']]
# kmeans = cluster.KMeans(n_clusters=2,init='random',n_init=1000, max_iter=1000,precompute_distances=True).fit(dataToKMeans)
# pd.DataFrame(kmeans.predict(dataToKMeans),columns=['status']).to_csv('data_csv/Clear/kmeans_result.csv',index=False)
#
# detectedData['status'] = pd.read_csv('data_csv/Clear/kmeans_result.csv')
# detectedData.to_csv('data_csv/Clear/detectedData_new.csv',index=False)

# detected = detected.drop(['Malmare 1','Malware 2','Malware 3'],axis=1)
detected['status'] = newDetectedData['status']
undetected['status'] = 2
allData = detected.append(undetected,ignore_index=True)
trainData, labelData = getTrainAndTestDataset(allData,android_general_permission)
allData_toTA = getTrainAndTestDataset(allData,android_general_permission)

X_train, X_test, y_train, y_test = model_selection.train_test_split(trainData, labelData, test_size=0.2)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

train_data_array = np.array(X_train)
train_data_array = train_data_array.reshape(train_data_array.shape[0],13,13,1)

test_data_array = np.array(X_test)
test_data_array = test_data_array.reshape(test_data_array.shape[0],13,13,1)

# test_data_array = np.array(X_test)
# test_data_array = test_data_array.reshape(test_data_array.shape[0],13,13,1)

# label_test = pd.DataFrame(np.random.randint(0,3,size=(100, 1)), columns=['label'])

#Model Baseline
model = keras.Sequential()

model.add(keras.layers.Conv2D(
    filters=3,
    kernel_size=3,
    strides = 1,
    activation='relu',
    input_shape=(13,13,1)))
model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))

model.add(keras.layers.Conv2D(
    filters=3,
    kernel_size=3,
    strides = 1,
    activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=120, activation='relu'))
model.add(keras.layers.Dense(units=84, activation='relu'))
model.add(keras.layers.Dense(units=3, activation='softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

checkpoint = keras.callbacks.ModelCheckpoint('model/KMeans_CNN_3500_clear/weights{epoch:08d}.h5', save_weights_only=True, period=100)

model.fit(train_data_array,
          keras.utils.to_categorical(y_train),
          steps_per_epoch = 10,
          epochs = 3500,
          callbacks = [checkpoint],
          validation_data=(test_data_array,keras.utils.to_categorical(y_test)),
          validation_steps= 10)
model.save('model/KMeans_CNN_3500_clear/model_TA_03.h5')
