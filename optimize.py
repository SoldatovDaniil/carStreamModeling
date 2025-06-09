import os
import system
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties
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
    return


def getTitle(type, r, g, lm):
    if type == "poisson":
        resStr = f"$\lambda$1={lm[0]}, $\lambda$2={lm[1]} п. Пуассона"
    else:
        resStr = f"$\lambda$1={lm[0]}, $\lambda$2={lm[1]}, r1={r[0]}, r2={r[1]}, g1={g[0]}, g2={g[1]} п. Бартлетта"
    return resStr


def tableManager(df, minIndex, areaInd, noStArea, iterDir, iterName, title):
    os.makedirs(iterDir, exist_ok=True)

    n_rows, n_cols = df.shape
    cell_text = []
    for idx, row in df.iterrows():
        formatted_row = [f"{val:.2f}" if val != -1 else "" for val in row]
        cell_text.append([idx] + formatted_row)
    col_labels = ['T4\T1'] + df.columns.tolist()
    colors = np.full((n_rows, n_cols + 1), 'white') 
    colors[0, :] = '#f0f0f0'
    colors[:, 0] = '#f0f0f0'
    for ind in areaInd:
        colors[ind[0], ind[1] + 1] = '#49f733ff'
    for ind in noStArea:
        colors[ind[0], ind[1] + 1] = 'gray'
    colors[minIndex[0], minIndex[1] + 1] = 'green'

    '''cell_height = 0.2
    cell_width = 0.6   
    fig_width = max(6, num_cols * cell_width)  
    fig_height = max(4, num_rows * cell_height) 
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    
    font = FontProperties(family='monospace', weight='bold')

    table = ax.table(cellText=cell_text, colLabels=col_labels, cellColours=colors.tolist(), loc='center', cellLoc='center', colWidths=[0.2] + [0.15]*num_cols)
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    for i in range(num_rows + 1):
        table[(i, 0)].set_height(0.3)
        for j in range(1, num_cols + 1):
            table[(i, j)].set_height(0.3)'''
    
    cell_height = max(0.1, min(0.5, 30 / n_rows))  
    cell_width = max(0.1, min(0.5, 30 / n_cols))   
    
    fig_width = cell_width * (n_cols + 2)  
    fig_height = cell_height * (n_rows + 4) 
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=100)
    
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 10], hspace=0.05)
    
    title_ax = fig.add_subplot(gs[0])
    title_ax.axis('off')
    title_ax.text(0.5, 0.2, title, ha='center', va='bottom', fontsize=min(16, max(10, 300 / n_cols)), fontweight='bold')
    
    ax = fig.add_subplot(111)
    ax.axis('off')

    table = ax.table(cellText=cell_text, colLabels=col_labels, cellColours=colors.tolist(), loc='center', cellLoc='center')

    font_size = max(6, min(12, 300 / max(n_rows, n_cols)))
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    
    for cell in table._cells.values():
        cell.set_height(cell_height / fig_height * 0.9)
        cell.PAD = 0.05

    outputPath = os.path.join(iterDir, f"T_{iterName}.png")
    plt.savefig(outputPath, dpi=300, bbox_inches='tight')
    plt.close()
    return


def graphicManager(x, y, minIndex, areaInd, noStArea, iterDir, iterName, title):
    os.makedirs(iterDir, exist_ok=True)
    
    xx, yy = np.meshgrid(x, y)

    mask1 = np.full((len(x), len(y)), False)
    for i,j in areaInd:
        mask1[i][j] = True
    mask2 = np.full((len(x), len(y)), False)
    for i,j in noStArea:
        mask2[i][j] = True
    mask3 = np.full((len(x), len(y)), False)
    mask3[minIndex[0], minIndex[1]] = True

    minSize, maxSize = 200, 500
    numPoints = len(x) * len(y)
    pointSize = max(minSize, min(maxSize, 10000/numPoints))

    plt.figure(figsize=(15, 15))
    plt.grid(True, linestyle=':', alpha=0.4, color='gray')
    plt.xticks(x)
    plt.yticks(y)
    plt.xlabel("T1")
    plt.ylabel("T4")
    plt.title(f"График средне взешанных оценок при:\n{title}")
    plt.scatter(xx, yy, s=pointSize, c='lightblue', alpha=0.7, edgecolors='gray')
    plt.scatter(xx[mask1], yy[mask1], s=pointSize, c='lightgreen', alpha=0.7, edgecolors='black')
    plt.scatter(xx[mask3], yy[mask3], s=pointSize, c='green', edgecolors='black')
    plt.scatter(xx[mask2], yy[mask2], s=pointSize, c='gray', alpha=0.7, edgecolors='black')

    outputPath = os.path.join(iterDir, f"GR_{iterName}.png")
    plt.savefig(outputPath, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    return


def findOptArea(arr, threshold):
    area = []
    minInd = np.unravel_index(np.argmin(np.where(np.isnan(arr), np.inf, arr)), arr.shape)
    minVal = arr[minInd[0], minInd[1]]
    mask = arr <= (minVal + threshold)
    inds = np.where(mask)
    area = list(zip(inds[0], inds[1]))
    return area, minVal, minInd


def findNoStacArea(arr):
    area = []
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):  
            if np.isnan(arr[i, j]):  
                area.append((i, j))
    return area


def visualisationResults(r, g, lm, gamma, systemInfo, T1, T4, type, threshold=1, iterDir="D:\\Diplom\\Experiments\\", iterName="testSave"):
    '''queque1 = [q[-2][0] if q != None else np.nan for q in systemInfo]
    queque1 = np.array(queque1).reshape((len(T4), len(T1)))

    queque2 = [q[-2][1] if q != None else np.nan for q in systemInfo]
    queque2 = np.array(queque2).reshape((len(T4), len(T1)))

    dfQ1 = pd.DataFrame(queque1, index=T4, columns=T1).fillna(-1)
    dfQ2 = pd.DataFrame(queque2, index=T4, columns=T1).fillna(-1)'''

    gammaWeight = []
    #gamma1 = []
    #gamma2 = []
    for ga in gamma:
        if None in ga:
            gammaWeight.append(np.nan)
            #gamma1.append(np.nan)
            #gamma2.append(np.nan)
        else:
            gammaWeight.append((lm[0] * ga[0] + lm[1] * ga[1]) / (lm[0] + lm[1]))
            #gamma1.append(g[0])
            #gamma2.append(g[1])

    gammaWeight = np.array(gammaWeight).reshape((len(T4), len(T1)))
    noStacArea = findNoStacArea(gammaWeight)
    optAreaIndexes, minG, minGIndex = findOptArea(gammaWeight, threshold)
    #gamma1 = np.array(gamma1).reshape((len(T4), len(T1)))
    #gamma2 = np.array(gamma2).reshape((len(T4), len(T1)))
    
    #dfG1 = pd.DataFrame(gamma1, index=T4, columns=T1).fillna(-1)
    #dfG2 = pd.DataFrame(gamma2, index=T4, columns=T1).fillna(-1)
    dfGammaWeight = pd.DataFrame(gammaWeight, index=T4, columns=T1).fillna(-1)
    title = getTitle(type, r, g, lm)
    graphicManager(T4, T1, minGIndex, optAreaIndexes, noStacArea, iterDir, iterName, title)
    tableManager(dfGammaWeight, minGIndex, optAreaIndexes, noStacArea, iterDir, iterName, title)

    #plt.figure(figsize=(8, 4))
    #sns.heatmap(dfGammaWeight, annot=True, fmt=".2f", cmap=["#ADD8E6"], cbar=False,linewidths=0.5, linecolor="black")
    #plt.title("Взвешенная оценка средних задержек")
    #plt.show()
    return minG, minGIndex


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
    return 


def optimizer(systemParams, T1Bounds, T4Bounds, stateTime, flagS, step = 1, EPS = 1, MaxSumTime = 100):
    T1Array = np.arange(T1Bounds[0], T1Bounds[1] + step, step)
    T4Array = np.arange(T4Bounds[0], T4Bounds[1] + step, step)
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


def onePairofLmOptimeze(lm, r, g, type, q, si, numberOfServiceStates, nSt, stateTime, maxSumTimeOfState, tBounds1, tBounds2,
                       eps=1, timeStep=1, flagForS=True, threshold=1):
    dir = f"D:\\Diplom\\Experiments\\r1={r[0]}_g1={g[0]}_r2={r[1]}_g2={g[1]}_sec={stateTime[1]}"
    iterName = f"lm1={lm[0]}_lm2={lm[1]}"
    resGamma = []
    resOptT = []
    print("=" * 20)
    print(f"Параметры: Lm1 = {lm[1]}, Lm2 = {lm[1]}, r1={r[0]}, r2={r[1]}, g1 = {g[0]}, g2={g[1]}")
    systemParams = [lm, type, r, g, q, si, numberOfServiceStates, nSt]
    gamma, grid, T1, T4, systemInfo = optimizer(systemParams, tBounds1, tBounds2, stateTime, flagForS, timeStep, eps, maxSumTimeOfState)
    MinG, MinGInd = visualisationResults(r, g, lm, gamma, systemInfo, T1, T4, type[0], threshold, dir, iterName)
    return MinG, T1[MinGInd[0]], T4[MinGInd[1]]


def BigExperiment(r, g, type, q, si, numberOfServiceStates, nSt, stateTime, maxSumTimeOfState, tBounds1, tBounds2,
               eps=1, timeStep=1, flagForS=True, threshold=1,
               lambdaBounds=[[0.1, 0.6], [0.1, 0.6]], lmstep=0.1):
    iterCount = 0
    lmArr1 = np.arange(lambdaBounds[0][0], lambdaBounds[0][1] + lmstep, lmstep)
    lmArr2 = np.arange(lambdaBounds[1][0], lambdaBounds[1][1] + lmstep, lmstep)
    lmArr1 = np.round(lmArr1, 2)
    lmArr2 = np.round(lmArr2, 2)
    dir = f"D:\\Diplom\\Experiments\\r1={r[0]}_g1={g[0]}_r2={r[1]}_g2={g[1]}"
    resGamma = []
    resOptT = []
    for lm1 in lmArr1:
        for lm2 in lmArr2:
            iterCount += 1
            print("=" * 20)
            print(f"Итерация {iterCount} из {len(lmArr1) * len(lmArr2)} Параметры: Lm1 = {lm1}, Lm2 = {lm2}")
            iterName = f"lm1={lm1}_lm2={lm2}"
            LM = [lm1, lm2]
            systemParams = [LM, type, r, g, q, si, numberOfServiceStates, nSt]
            gamma, grid, T1, T4, systemInfo = optimizer(systemParams, tBounds1, tBounds2, stateTime, flagForS, timeStep, eps, maxSumTimeOfState)
            iterMinG, iterMinGInd = visualisationResults(gamma, systemInfo, T1, T4, threshold, dir, iterName)
            resGamma.append(iterMinG)
            resOptT.append((T1[iterMinGInd[0]], T4[iterMinGInd[1]]))

    resGamma = np.array(resGamma).reshape((len(lmArr2), len(lmArr1)))
    dfRes = pd.DataFrame(resGamma, index=lmArr2, columns=lmArr1)
    
    num_rows, num_cols = dfRes.shape
    cell_text = []
    for idx, row in dfRes.iterrows():
        formatted_row = [f"{val:.2f}" for val in row]
        cell_text.append([idx] + formatted_row)  
    col_labels = [''] + dfRes.columns.tolist()
    cell_height = 0.2
    cell_width = 0.6   
    fig_width = max(6, num_cols * cell_width)  
    fig_height = max(4, num_rows * cell_height) 
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    font = FontProperties(family='monospace', weight='bold')
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center', colWidths=[0.2] + [0.15]*num_cols)
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    for i in range(num_rows + 1):
        table[(i, 0)].set_height(0.3)
        for j in range(1, num_cols + 1):
            table[(i, j)].set_height(0.3)
    outputPath =  f"D:\\PyProjects\\Experiments\\r={r[0]}_g={g[0]}\T1_r={r[0]}_g={g[0]}.png"
    plt.savefig(outputPath, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    newResOptT = []
    tmp = []
    count = 0
    for i in resOptT:
        count += 1
        tmp.append(i)
        if count % len(lmArr1) == 0:
            newResOptT.append(tmp)
            tmp = []
    dfResOptT = pd.DataFrame(newResOptT, index=lmArr2, columns=lmArr1)
    dfResOptT.to_excel(f"D:\\PyProjects\\Experiments\\r={r[0]}_g={g[0]}\T2_r={r[0]}_g={g[0]}.xlsx", index=True)
    return resOptT, resGamma


def sameLmExperiment(lmbda, r, g, type, q, si, numberOfServiceStates, nSt, stateTime, maxSumTimeOfState, tBounds1, tBounds2,
                       eps=1, timeStep=1, flagForS=True, threshold=1):
    iterCount = 0
    for lm in lmbda:
        iterCount += 1
        print("+" * 20)
        onePairofLmOptimeze([lm, lm], r, g, type, q, si, numberOfServiceStates, nSt, stateTime, maxSumTimeOfState, tBounds1, tBounds2, eps, timeStep, flagForS, threshold)
        print(f"Посчитано {iterCount} из {len(lmbda)}")
        

Lambda = [0.4, 0.4]
Type = ['poisson', 'poisson']
R = [0.5, 0.5]
G = [0.5, 0.5]
Q = [[], []]
SI = [[1, 0.5], [1, 0.5]]
NumberOfServiceStates = [[1, 2], [4, 5]]
Nst = 1000
StateTime= [10, 0, 3, 10, 0, 3]
MaxSumTimeOfStates = 150
Eps = 0.5
StepTime = 1
FlagForS = False # Проверять ли близость оценок дисперсий(False - да, True - нет)
Threshold = 1 # Для построение квазиоптимальной области

#testSys = system.System(Lambda, Type, R, G, Q, SI, NumberOfServiceStates, Nst, StateTime)
#print(testSys.processing())

#systemWork(testSys, FlagForS)
#visualisationDynamics(testSys.getSystemDynamics(), testSys.getTimeArray(), testSys.getSystemQChanges())

#gamma, grid, T1, T4, systemInfo = optimizer([Lambda, Type, R, G, Q, SI, NumberOfServiceStates, Nst], [30, 35], [30, 35], StateTime, FlagForS, StepTime, Eps, MaxSumTimeOfStates)
#visualisationResults(gamma, systemInfo, T1, T4)

#experiment(R, G, Type, Q, SI, NumberOfServiceStates, Nst, StateTime, 
#           MaxSumTimeOfStates, [20, 25], [20, 25], Eps, StepTime, FlagForS, Threshold, [[0.1, 0.2], [0.1, 0.2]], 0.1)

onePairofLmOptimeze(Lambda, R, G, Type, Q, SI, NumberOfServiceStates, Nst, StateTime, 
                   MaxSumTimeOfStates, [3, 50], [3, 50], Eps, StepTime, FlagForS, Threshold)

'''
x = np.arange(1, 50, 1)
y = np.arange(1, 50, 1)
minIndex = [0, 0]
area_indices = np.random.choice(400, 20, replace=False)  # 20 случайных индексов
area_indices = [(idx//20, idx%20) for idx in area_indices]
noStArea = [[0,0]]
iterDir = "D:\\Diplom\\Experiments\\"
iterName = "Test"
title = "123 123"
graphicManager(x, y, minIndex, area_indices, noStArea, iterDir, iterName, title)
'''

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
