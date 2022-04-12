import glob
import random

imgs_listFull = glob.glob(
    "C:/Users/Pang Bo/Desktop/大三下/方言/Spoken-language-identification-pytorch/images/speaker01/*.png")  # 返回所匹配的文件名列表
imgs_list = [i[len("C:/Users/Pang Bo/Desktop/大三下/方言/Spoken-language-identification-pytorch/images/speaker01"):] for i in
             imgs_listFull]
train_per = 0.8
valid_per = 0.0
test_per = 0.2

random.seed(666)
random.shuffle(imgs_list)

imgs_num = len(imgs_list)
train_point = int(imgs_num * train_per)
valid_point = int(imgs_num * valid_per)

trainList = imgs_list[0:train_point]
valList = imgs_list[train_point:train_point + valid_point]
testList = imgs_list[train_point + valid_point:imgs_num - 1]

fileTrainIRRG = open("C:/Users/Pang Bo/Desktop/大三下/方言/Spoken-language-identification-pytorch/txtForTrain/trainIRRG.txt",
                     'w')
fileValIRRG = open("C:/Users/Pang Bo/Desktop/大三下/方言/Spoken-language-identification-pytorch/txtForTrain/valIRRG.txt", 'w')
fileTestIRRG = open("C:/Users/Pang Bo/Desktop/大三下/方言/Spoken-language-identification-pytorch/txtForTrain/testIRRG.txt",
                    'w')

for line in trainList:
    line2 = line + ' ' + line.replace("IRRG", "label")
    fileTrainIRRG.write(line2)
    fileTrainIRRG.write('\n')

for line in valList:
    line2 = line + ' ' + line.replace("IRRG", "label")
    fileValIRRG.write(line)
    fileValIRRG.write('\n')

for line in testList:
    line2 = line + ' ' + line.replace("IRRG", "label")
    fileTestIRRG.write(line2)
    fileTestIRRG.write('\n')

fileTrainIRRG.close()
fileValIRRG.close()
fileTestIRRG.close()