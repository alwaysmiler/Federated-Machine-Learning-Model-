import tensorflow as tf
import csv
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

datafolder=r'C:\Users\tingx\Downloads\FirstMeasurement\AllData'

ResultFLUO50=[]
ResultFLUO75=[]
ResultFLUO100=[]
ResultVIS50=[]
ResultVIS75=[]
ResultVIS100=[]
list1=['ResultFLUO-', 'ResultVIS-']
list2=[50,75,100]
sample1=[1,2,3,4,5]
sample2=[1,2,3,4,5,6,7,8,9,10]
meas=[1,2,3,4]

for Res in list1:
    for Portion in list2:
        tempDatalist=[]
        if Portion==100:
            for sam in sample2:
                for mea in meas:
                    filenamestring=Res + str(Portion) + 'S' + str(sam) + 'M' + str(mea) + '.csv'
                    csvfile = os.path.join(datafolder, filenamestring)
                    #print(csvfile)
                    with open(csvfile, newline='') as f:
                        templist = []
                        reader = csv.reader(f)
                        data = list(reader)
                        data = data[1:]
                        # data2=[elem[2:] for elem in data]
                        for elem in data:
                            templist.append(float(elem[2]))
                        tempDatalist.append(templist)

        else:
            for sam in sample1:
                for mea in meas:
                    filenamestring = Res + str(Portion) + 'S' + str(sam) + 'M' + str(mea) + '.csv'
                    csvfile = os.path.join(datafolder, filenamestring)
                    with open(csvfile, newline='') as f:

                        templist = []
                        reader = csv.reader(f)
                        data = list(reader)
                        data = data[1:]
                        # data2=[elem[2:] for elem in data]
                        for elem in data:
                            templist.append(float(elem[2]))
                        #if Res == 'ResultFLUO-' and Portion == 50:
                            #print(filenamestring)
                            #print(templist)
                        tempDatalist.append(templist)

        if Res=='ResultFLUO-' and Portion==50:
            ResultFLUO50 = tempDatalist
        if Res=='ResultFLUO-' and Portion==75:
            ResultFLUO75 = tempDatalist
        if Res=='ResultFLUO-' and Portion==100:
            ResultFLUO100 = tempDatalist
        if Res=='ResultVIS-' and Portion==50:
            ResultVIS50 = tempDatalist
        if Res=='ResultVIS-' and Portion==75:
            ResultVIS75 = tempDatalist
        if Res=='ResultVIS-' and Portion==100:
            ResultVIS100 = tempDatalist

splitratio=2/3
x_trainlist=ResultFLUO50[:int(len(ResultFLUO50)*splitratio)]+ResultFLUO75[:int(len(ResultFLUO75)*splitratio)]+ResultFLUO100[:int(len(ResultFLUO100)*splitratio)]
y_trainlist=[0]*int(len(ResultFLUO50)*splitratio)+[1]*int(len(ResultFLUO75)*splitratio)+[2]*int(len(ResultFLUO100)*splitratio)
x_testlist=ResultFLUO50[int(len(ResultFLUO50)*splitratio):]+ResultFLUO75[int(len(ResultFLUO75)*splitratio):]+ResultFLUO100[int(len(ResultFLUO100)*splitratio):]
y_testlist=[0]*(len(ResultFLUO50)-int(len(ResultFLUO50)*splitratio))+[1]*(len(ResultFLUO75)-int(len(ResultFLUO75)*splitratio))+[2]*(len(ResultFLUO100)-int(len(ResultFLUO100)*splitratio))

print(int(len(ResultFLUO50)*splitratio))
print(int(len(ResultFLUO75)*splitratio))
print(int(len(ResultFLUO100)*splitratio))
print(len(ResultFLUO50)-int(len(ResultFLUO50)*splitratio))
print(len(ResultFLUO75)-int(len(ResultFLUO75)*splitratio))
print(len(ResultFLUO100)-int(len(ResultFLUO100)*splitratio))


x_train=np.asarray(x_trainlist)
y_train=np.asarray(y_trainlist)
x_test=np.asarray(x_testlist)
y_test=np.asarray(y_testlist)

print(y_test)
model = tf.keras.models.Sequential([
  #tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=30)
model.evaluate(x_test,  y_test, verbose=2)
print(" Model Prediction")
#print((x_test[5]).shape)
print(model.predict(x_test))
