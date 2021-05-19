
import math


file = open('TrainingSet.csv', 'r')
training = file.readlines()
for i in range(0, len(training)):
    training[i] = training[i][0:-1]
    training[i] = training[i].split(',')

MeanHM = 0.0
MeanHW = 0.0
MeanWM = 0.0
MeanWW = 0.0
MeanAM = 0.0
MeanAW = 0.0

VarHM = 0.0
VarHW = 0.0
VarWM = 0.0
VarWW = 0.0
VarAM = 0.0
VarAW = 0.0

TM = 0.0
TW = 0.0

for i in range(0, len(training)):
    if training[i][3] == 'M':
        TM += 1
        MeanHM += int(training[i][0])
        MeanWM += int(training[i][1])
        MeanAM += int(training[i][1])
    else:
        TW += 1
        MeanHW += int(training[i][0])
        MeanWW += int(training[i][1])
        MeanAW += int(training[i][1])

PM = TM / len(training)
PW = TW / len(training)

MeanHM = MeanHM / TM
MeanHW = MeanHW / TW
MeanWM = MeanWM / TM
MeanWW = MeanWW / TW
MeanAM = MeanAM / TM
MeanAW = MeanAW / TW


for i in range(0, len(training)):
    if training[i][3] == 'M':
        VarHM += pow((int(training[i][0]) - MeanHM), 2)
        VarWM += pow((int(training[i][1]) - MeanWM), 2)
        VarAM += pow((int(training[i][2]) - MeanAM), 2)
    else:
        VarHW += pow((int(training[i][0]) - MeanHW), 2)
        VarWW += pow((int(training[i][1]) - MeanWW), 2)
        VarAW += pow((int(training[i][2]) - MeanAW), 2)

VarHM /= (TM-1)
VarWM /= (TM-1)
VarAM /= (TM-1)
VarHW /= (TW-1)
VarWW /= (TW-1)
VarAW /= (TW-1)


CHM = (1 / math.sqrt(2*math.pi*VarHM))
CWM = (1 / math.sqrt(2*math.pi*VarWM))
CAM = (1 / math.sqrt(2*math.pi*VarAM))
CHW = (1 / math.sqrt(2*math.pi*VarHW))
CWW = (1 / math.sqrt(2*math.pi*VarWW))
CAW = (1 / math.sqrt(2*math.pi*VarAW))


CommonM = math.log(PM, math.e)+math.log(CHM, math.e) + \
    math.log(CWM, math.e)+math.log(CAM, math.e)
CommonW = math.log(PW, math.e)+math.log(CHW, math.e) + \
    math.log(CWW, math.e)+math.log(CAW, math.e)


condition = True
while condition:
    M = 0
    W = 0
    case = input("1: Gaussian Naive Bayes")
    if case == '1':
        print("Enter data for prediction.")
        h = int(input("Height : "))
        w = int(input("Weight : "))
        a = int(input("Age : "))

        exp = (-0.5)*((pow((h-MeanHM), 2)/VarHM) +
                      (pow((w-MeanWM), 2)/VarWM) + (pow((a-MeanAM), 2)/VarAM))
        ProbM = CommonM+exp

        exp = (-0.5)*((pow((h-MeanHW), 2)/VarHW) +
                      (pow((w-MeanWW), 2)/VarWW) + (pow((a-MeanAW), 2)/VarAW))
        ProbW = CommonW+exp

        if ProbM > ProbW:
            print("\n\nClass : M\n\n")
        else:
            print("\n\nClass : W\n\n")
    else:
        print("Incorrect choice.")
