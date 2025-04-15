import system
import numpy as np
from matplotlib import pyplot as plt


def systemWork(sys : system.System, EPS = 1):
    gammaPrev1 = 0
    gammaPrev2 = 0
    gammaNext1 = -10
    gammaNext2 = -10
    EpochsCounter = 0
    while abs(gammaPrev1 - gammaNext1) > EPS and abs(gammaPrev2 - gammaNext2) > EPS:
        gammaPrev1, gammaPrev2 = gammaNext1, gammaNext2
        systemInfo = sys.processing()
        gammaNext1, gammaNext2 = systemInfo[0], systemInfo[1]
        EpochsCounter += 1
        if systemInfo[-1] == False:
            #print("Нет стационара")
            return systemInfo
        sys.setNstop = sys.getNstop() * 2
        #print(f"Эпох пройдено: {EpochsCounter}", end=' ')
    return systemInfo


def optimizer(systemParams, T1Bounds, T4Bounds, stateTime, step = 1, EPS = 1):
    T1Array = np.arange(T1Bounds[0], T1Bounds[1], step)
    T4Array = np.arange(T4Bounds[0], T4Bounds[1], step)
    gamma = []
    grid = []
    systemInfoArray = []
    iterCounter = 0
    for T1 in T1Array:
        for T4 in T4Array:
            stateTime[0] = T1
            stateTime[3] = T4
            sys = system.System(*systemParams, stateTime)
            systemInfo = systemWork(sys, EPS)
            systemInfoArray.append(systemInfo)
            if systemInfo[-1] == False:
                gamma.append(False)
            else:
                gamma.append((systemInfo[0], systemInfo[1]))
            grid.append((T1, T4))
            iterCounter += 1
            print(f"Точка {iterCounter} из {len(T1Array) * len(T4Array)}")    
    return gamma, grid, T1Array, T4Array, systemInfoArray


Lambda = [0.2, 0.2]
Type = ['poisson', 'poisson']
R = [0.6, 0.6]
G = [0.3, 0.3]
Q = [[], []]
SI = [[1, 2], [1, 2]]
Nst = 1000
StateTime= [15, 3, 3, 15, 3, 3]
#testSys = system.System(Lambda, Type, R, G, Q, SI, Nst, StateTime)
gamma, grid, T1, T4, systemInfo = optimizer([Lambda, Type, R, G, Q, SI, Nst], [5, 50], [5, 50], StateTime)

queque = [sum(q[-2]) for q in systemInfo]
queque = np.array(queque).reshape((len(T1), len(T4)))

gammaWeight = []
for g in gamma:
    if g == False:
        gammaWeight.append(0)
    else:
        gammaWeight.append((Lambda[0] * g[0] + Lambda[1] * g[1]) / (Lambda[0] * Lambda[1]))
gammaWeight = np.array(gammaWeight).reshape((len(T1), len(T4)))

plt.pcolormesh(T1, T4, gammaWeight, shading='auto', cmap='viridis')
plt.colorbar(label='gammaWeight')
plt.show()

plt.pcolormesh(T1, T4, queque, shading='auto', cmap='viridis')
plt.colorbar(label='queque')
plt.show()
