import system
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import sys


def systemWork(sys : system.System, flagForS = True, EPS = 1):
    gammaPrev1 = 0
    gammaPrev2 = 0
    sPrev1 = 0
    sPrev2 = 0
    sNext1 = -10
    sNext2 = -10
    gammaNext1 = -10
    gammaNext2 = -10
    EpochsCounter = 0
    while (abs(gammaPrev1 - gammaNext1) > EPS or abs(gammaPrev2 - gammaNext2) > EPS) \
        and ((abs(sPrev1 - sNext1) > EPS or abs(sPrev2 - sNext2) > EPS) or flagForS):
        gammaPrev1, gammaPrev2 = gammaNext1, gammaNext2
        sPrev1, sPrev2 = sNext1, sNext2
        systemInfo = sys.processing()
        gammaNext1, gammaNext2 = systemInfo[0], systemInfo[1]
        sNext1, sNext2 = systemInfo[2], systemInfo[3]
        EpochsCounter += 1
        if systemInfo[-1] == False:
            print("Нет стационара")
            return systemInfo
        sys.setNstop(sys.getNstop() + 1000)
        #print(gammaNext1, gammaNext2, sNext1, sNext2)
    print(f"Эпох пройдено = {EpochsCounter}", f"Порог количества обслужанных машин = {sys.getNstop() - 1000}")
    printSysteminfo(systemInfo)
    return systemInfo


def printSysteminfo(info):
    valuesNames = ["gamma1", "gamma2", "s1", "s2", "Количество циклов", "Количество обслуженных машин 1-го потока", 
                "2-го потока", "Количество поступивших машин 1-го потока", "2-го потока", "Конечные очереди"]
    counter = 0
    if info[-1] == False:
        print("Нет стационара")
    for vn, value in zip(valuesNames, info[:-1]):
        print(f"{vn} = {value} ", end=' ')
        counter += 1
        if counter == 4 or counter == 5 or counter == 7 or counter == 9:
            print()
    print()


def optimizer(systemParams, T1Bounds, T4Bounds, stateTime, flagS, step = 1, EPS = 1, MaxSumTime = 100):
    T1Array = np.arange(T1Bounds[0], T1Bounds[1], step)
    T4Array = np.arange(T4Bounds[0], T4Bounds[1], step)
    gamma = []
    grid = []
    systemInfoArray = []
    iterCounter = 0
    for T1 in T1Array:
        for T4 in T4Array:
            iterCounter += 1
            print("-" * 20)
            print(f"Точка {iterCounter} из {len(T1Array) * len(T4Array)} Параметры: T1 = {T1}, T4 = {T4}") 
            if T1 + T4 > MaxSumTime:
                gamma.append((None, None))
                systemInfoArray.append(None)
                print(f"Сумма T1 + T4 > {MaxSumTime}")
                continue

            stateTime[0] = T1
            stateTime[3] = T4
            sys = system.System(*systemParams, stateTime)
            systemInfo = systemWork(sys, flagS, EPS)
            systemInfoArray.append(systemInfo)
            if systemInfo[-1] == False:
                gamma.append((None, None))
            else:
                gamma.append((systemInfo[0], systemInfo[1]))
            grid.append((T1, T4))

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
    dfGammaWeight = pd.DataFrame(gammaWeight, index=T4, columns=T1).fillna(-1)
    
    plt.figure(figsize=(8, 4))
    sns.heatmap(dfGammaWeight, annot=True, fmt=".1f",cmap="YlGnBu",cbar=False)
    plt.title("Взвешенная оценка средних задержек")
    plt.show()
    '''
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
    '''

    dfQ1.to_csv('Q1.csv')
    dfQ2.to_csv('Q2.csv')
    dfG1.to_csv('G1.csv')
    dfG2.to_csv('G2.csv')
    dfGammaWeight.to_csv('GW.csv')


def visualisationDynamics(systemDynamic, t, systemQ):
    dynG, dynS = systemDynamic
    dynq1, dynq2 = systemQ
    dynG1 = [dG[0] for dG in dynG]
    dynG2 = [dG[1] for dG in dynG]
    dynS1 = [dS[0] for dS in dynS]
    dynS2 = [dS[1] for dS in dynS]
    t1 = [dt1[0] for dt1 in dynq1]
    t2 = [dt2[0] for dt2 in dynq2]
    q1 = [dq1[1] for dq1 in dynq1]
    q2 = [dq2[1] for dq2 in dynq2]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))

    axes[0, 0].plot(t, dynG1, label='g1', color="red")
    axes[0, 0].plot(t, dynG2, label='g2', color="blue")
    axes[0, 0].set_title("Динамика gamma")

    axes[0, 1].plot(t, dynS1, label='s1', color="red")
    axes[0, 1].plot(t, dynS2, label='s2', color="blue")
    axes[0, 1].set_title("Динамика s")

    axes[1, 0].step(t1, q1, where='post', color='red', linestyle='-')
    axes[1, 0].set_title("Динамика q1")

    axes[1, 1].step(t2, q2, where='post', color='blue', linestyle='-')
    axes[1, 1].set_title("Динамика q2")

    plt.tight_layout()
    plt.show()


Lambda = [0.3, 0.2]
Type = ['poisson', 'poisson']
R = [0.508, 0.546]
G = [0.508, 0.546]
Q = [[], []]
SI = [[1, 0.5], [1, 0.5]]
NumberOfServiceStates = [[1, 2], [4, 5]]
Nst = 1000
StateTime= [60, 3, 3, 10, 3, 3]
MaxSumTimeOfStates = 150
Eps = 1
StepTime = 1
FlagForS = True # Проверять ли близость оценок дисперсий(False - да, True - нет)
#testSys = system.System(Lambda, Type, R, G, Q, SI, NumberOfServiceStates, Nst, StateTime)
#print(testSys.processing())
#systemWork(testSys, FlagForS)
#visualisationDynamics(testSys.getSystemDynamics(), testSys.getTimeArray(), testSys.getSystemQChanges())
#gamma, grid, T1, T4, systemInfo = optimizer([Lambda, Type, R, G, Q, SI, NumberOfServiceStates, Nst], [30, 40], [5, 40], StateTime, FlagForS, StepTime, Eps, MaxSumTimeOfStates)
#visualisationResults(gamma, systemInfo, T1, T4)

'''
g1 = []
g2 = []
ques = []
time = []
for t in range(10, 150):
    StateTime[0] = t
    time.append(t)
    sys = system.System(Lambda, Type, R, G, Q, SI, NumberOfServiceStates, Nst, StateTime)
    systemInfo = systemWork(sys, FlagForS)
    ques.append(systemInfo[-2])
    g1.append(systemInfo[0])
    g2.append(systemInfo[1])

plt.plot(time, g1, color='red')
plt.plot(time, g2, color='green')
plt.show()

q1 = [q[0] for q in ques]
q2 = [q[1] for q in ques]
plt.plot(time, q1, color='red')
#plt.plot(time, q2, color='green')
plt.show()
'''
