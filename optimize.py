import system
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def systemWork(sys : system.System, EPS = 1):
    gammaPrev1 = 0
    gammaPrev2 = 0
    sPrev1 = 0
    sPrev2 = 0
    sNext1 = -10
    sNext2 = -10
    gammaNext1 = -10
    gammaNext2 = -10
    EpochsCounter = 0
    while (abs(gammaPrev1 - gammaNext1) > EPS and abs(gammaPrev2 - gammaNext2) > EPS) \
        and (abs(sPrev1 - sNext1) > EPS and abs(sPrev2 - sNext2) > EPS):
        gammaPrev1, gammaPrev2 = gammaNext1, gammaNext2
        sPrev1, sPrev2 = sNext1, sNext2
        systemInfo = sys.processing()
        gammaNext1, gammaNext2 = systemInfo[0], systemInfo[1]
        sNext1, sNext2 = systemInfo[2], systemInfo[3]
        EpochsCounter += 1
        if systemInfo[-1] == False:
            print("Нет стационара", end=' ')
            return systemInfo
        sys.setNstop(sys.getNstop() * 2)
    print(f"Эпох пройдено: {EpochsCounter}", f"Количесвто обслужанных машин: {sys.getNstop()}", end=' ')
    return systemInfo


def optimizer(systemParams, T1Bounds, T4Bounds, stateTime, step = 1, EPS = 1, MaxSumTime = 100):
    T1Array = np.arange(T1Bounds[0], T1Bounds[1], step)
    T4Array = np.arange(T4Bounds[0], T4Bounds[1], step)
    gamma = []
    grid = []
    systemInfoArray = []
    iterCounter = 0
    for T1 in T1Array:
        for T4 in T4Array:
            if T1 + T4 > MaxSumTime:
                gamma.append((None, None))
                systemInfoArray.append(None)
                continue
            stateTime[0] = T1
            stateTime[3] = T4
            sys = system.System(*systemParams, stateTime)
            systemInfo = systemWork(sys, EPS)
            systemInfoArray.append(systemInfo)
            if systemInfo[-1] == False:
                gamma.append((None, None))
            else:
                gamma.append((systemInfo[0], systemInfo[1]))
            grid.append((T1, T4))
            iterCounter += 1
            print(f"Точка {iterCounter} из {len(T1Array) * len(T4Array)}") 
    return gamma, grid, T1Array, T4Array, systemInfoArray


def visualisationResults(gamma, systemInfo, T1, T4):
    queque1 = [q[-2][0] if q != None else np.nan for q in systemInfo]
    queque1 = np.array(queque1).reshape((len(T4), len(T1)))

    queque2 = [q[-2][1] if q != None else np.nan for q in systemInfo]
    queque2 = np.array(queque2).reshape((len(T4), len(T1)))

    dfQ1 = pd.DataFrame(queque1, index=T4, columns=T1).fillna(-1)
    dfQ2 = pd.DataFrame(queque2, index=T4, columns=T1).fillna(-1)

    gammaWeight = []
    gamma1 = []
    gamma2 = []
    for g in gamma:
        if None in g:
            gammaWeight.append(np.nan)
            gamma1.append(np.nan)
            gamma2.append(np.nan)
        else:
            gammaWeight.append((Lambda[0] * g[0] + Lambda[1] * g[1]) / (Lambda[0] + Lambda[1]))
            gamma1.append(g[0])
            gamma2.append(g[1])
    gammaWeight = np.array(gammaWeight).reshape((len(T4), len(T1)))
    gamma1 = np.array(gamma1).reshape((len(T4), len(T1)))
    gamma2 = np.array(gamma2).reshape((len(T4), len(T1)))
    
    dfG1 = pd.DataFrame(gamma1, index=T4, columns=T1).fillna(-1)
    dfG2 = pd.DataFrame(gamma2, index=T4, columns=T1).fillna(-1)
    
    plt.pcolormesh(T1, T4, gammaWeight, shading='auto', cmap='viridis')
    plt.colorbar(label='gammaWeight')
    plt.show()

    plt.pcolormesh(T1, T4, gamma1, shading='auto', cmap='viridis')
    plt.colorbar(label='gamma1')
    plt.show()

    plt.pcolormesh(T1, T4, gamma2, shading='auto', cmap='viridis')
    plt.colorbar(label='gamma2')
    plt.show()

    plt.pcolormesh(T1, T4, queque1, shading='auto', cmap='viridis')
    plt.colorbar(label='queque1')
    plt.show()

    plt.pcolormesh(T1, T4, queque2, shading='auto', cmap='viridis')
    plt.colorbar(label='queque2')
    plt.show()

    dfQ1.to_csv('Q1.csv')
    dfQ2.to_csv('Q2.csv')
    dfG1.to_csv('G1.csv')
    dfG2.to_csv('G2.csv')
    return

Lambda = [0.3, 0.3]
Type = ['poisson', 'poisson']
R = [0.6, 0.6]
G = [0.3, 0.3]
Q = [[], []]
SI = [[1, 0.5], [1, 0.5]]
NumberOfServiceStates = [[1, 2], [4, 5]]
Nst = 1000
StateTime= [15, 3, 3, 15, 3, 3]
MaxSumTimeOfStates = 90
Eps = 1
StepTime = 1
#testSys = system.System(Lambda, Type, R, G, Q, SI, NumberOfServiceStates, Nst, StateTime)
#print(testSys.processing())
#print(systemWork(testSys))
gamma, grid, T1, T4, systemInfo = optimizer([Lambda, Type, R, G, Q, SI, NumberOfServiceStates, Nst], [29, 30], [29, 30], StateTime, StepTime, Eps, MaxSumTimeOfStates)
visualisationResults(gamma, systemInfo, T1, T4)

'''qChanges = testSys.getSystemQChanges()
TimeMoments1 = [q[0] for q in qChanges[0]]
Q1 = [q[1] for q in qChanges[0]]
TimeMoments2 = [q[0] for q in qChanges[1]]
Q2 = [q[1] for q in qChanges[1]]

plt.plot(TimeMoments1, Q1)
plt.show()'''

'''g1 = []
g2 = []
ques = []
time = []
for t in range(10, 150):
    StateTime[0] = t
    time.append(t)
    sys = system.System(Lambda, Type, R, G, Q, SI, NumberOfServiceStates, Nst, StateTime)
    systemInfo = systemWork(sys)
    ques.append(systemInfo[-2])
    g1.append(systemInfo[0])
    g2.append(systemInfo[1])

plt.plot(time, g1, color='red')
#plt.plot(time, g2, color='green')
plt.show()

q1 = [q[0] for q in ques]
q2 = [q[1] for q in ques]
plt.plot(time, q1, color='red')
#plt.plot(time, q2, color='green')
plt.show()'''
