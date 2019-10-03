import pandas as pd
import numpy as np
import csv
import os
from pathlib import Path
from tensorflow import keras
from androguard import misc
listOfInputApkPermission = [['File Name','Permission']]
x=1
for filename in os.listdir('/home/jhonarox/Documents/TA-D4TI03/APK Test/Danger less'):
    print(x)
    x+=1
    try:
        a, d, dx = misc.AnalyzeAPK("/home/jhonarox/Documents/TA-D4TI03/Koodous/APKs/Undetected/1a6059c50cba6906de18ce0348e58678e79bf4cc0a5ce1b28e7b40ed771dca34.apk")
        print(a.get_app_name())
        input_apk_permission = a.get_permissions()
        listOfInputApkPermission.append((a.get_app_name,','.join(input_apk_permission)))
    except :
        continue

headers = listOfInputApkPermission.pop(0)
df = pd.DataFrame(listOfInputApkPermission, columns=headers)
# df.to_csv('Detected.csv')
print(df)

data_test_detected = getTrainAndTestDataset(df,android_general_permission)
data_test_detected = data_test_detected.astype('float32')
data_test_detected = np.array(data_test_detected)
data_test_detected = data_test_detected.reshape(data_test_detected.shape[0],13,13,1)
load_model = keras.models.load_model('model/KMeans_CNN_6000/model_TA_03.h5')
result = load_model.predict_classes(data_test_detected)

if result == 0:
    print("Kurang Berbahaya")
elif result == 1:
    print("Berbahaya")
else:
    print("Tidak Berbahaya")


#/home/jhonarox/Documents/TA-D4TI03/APK Test/Dange