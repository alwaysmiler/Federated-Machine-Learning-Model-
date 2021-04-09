import tensorflow as tf
import csv
import numpy as np
import os
import warnings
import random
warnings.filterwarnings("ignore")

class TFML:


    def dataProcess(self):
        datafolder = r'C:\Users\tingx\Downloads\FirstMeasurement\AllData'

        ResultFLUO50 = []
        ResultFLUO75 = []
        ResultFLUO100 = []
        ResultVIS50 = []
        ResultVIS75 = []
        ResultVIS100 = []
        list1 = ['ResultFLUO-', 'ResultVIS-']
        list2 = [50, 75, 100]
        sample1 = [1, 2, 3, 4, 5]
        sample2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        meas = [1, 2, 3, 4]

        for Res in list1:
            for Portion in list2:
                tempDatalist = []
                if Portion == 100:
                    for sam in sample2:
                        for mea in meas:
                            filenamestring = Res + str(Portion) + 'S' + str(sam) + 'M' + str(mea) + '.csv'
                            csvfile = os.path.join(datafolder, filenamestring)
                            # print(csvfile)
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
                                # if Res == 'ResultFLUO-' and Portion == 50:
                                # print(filenamestring)
                                # print(templist)
                                tempDatalist.append(templist)

                if Res == 'ResultFLUO-' and Portion == 50:
                    ResultFLUO50 = tempDatalist
                if Res == 'ResultFLUO-' and Portion == 75:
                    ResultFLUO75 = tempDatalist
                if Res == 'ResultFLUO-' and Portion == 100:
                    ResultFLUO100 = tempDatalist
                if Res == 'ResultVIS-' and Portion == 50:
                    ResultVIS50 = tempDatalist
                if Res == 'ResultVIS-' and Portion == 75:
                    ResultVIS75 = tempDatalist
                if Res == 'ResultVIS-' and Portion == 100:
                    ResultVIS100 = tempDatalist

        splitratio = 3 / 8
        x_datalist = ResultFLUO50 + ResultFLUO75 + ResultFLUO100
        y_datalist = [0] * int(len(ResultFLUO50)) + [1] * int(len(ResultFLUO75)) + [2] * int(len(ResultFLUO100))

        Alldatalist = list(zip(x_datalist, y_datalist))
        random.shuffle(Alldatalist)
        x_datalist, y_datalist = zip(*Alldatalist)

        x_trainlist = x_datalist[:int(len(x_datalist) * splitratio)]
        y_trainlist = y_datalist[:int(len(y_datalist) * splitratio)]
        x_testlist = x_datalist[int(len(x_datalist) * splitratio):]
        y_testlist = y_datalist[int(len(y_datalist) * splitratio):]

        x_train = np.asarray(x_trainlist)
        y_train = np.asarray(y_trainlist)
        x_test = np.asarray(x_testlist)
        y_test = np.asarray(y_testlist)
        return x_train, y_train, x_test, y_test

    def __init__(self,name):
        self.name=name
        self.x_train,self.y_train,self.x_test, self.y_test=self.dataProcess()
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=self.x_train[0].shape),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    def run(self):
        self.model.fit(self.x_train, self.y_train, epochs=1)

    def eval(self):
        self.model.evaluate(self.x_test, self.y_test, verbose=2)
        modelPre = self.model.predict(self.x_test)
        #print(modelPre)
        pred = []
        for i in range(len(modelPre)):
            if modelPre[i][0] == 1.0:
                pred.append(0)
            elif modelPre[i][1] == 1.0:
                pred.append(1)
            else:
                pred.append(2)
        print("Y_testing")
        print(self.y_test)
        print("Y Prediction")
        print(np.asarray(pred))
        print("Accuracy")
        self.model.evaluate(self.x_test, self.y_test, verbose=2)












