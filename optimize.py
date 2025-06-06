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


def tableManager(df, areaInd, iterDir, iterName):
    os.makedirs(iterDir, exist_ok=True)

    num_rows, num_cols = df.shape
    cell_text = []
    for idx, row in df.iterrows():
        formatted_row = [f"{val:.2f}" for val in row]
        cell_text.append([idx] + formatted_row)  
    col_labels = [''] + df.columns.tolist()
    colors = np.full((num_rows, num_cols + 1), 'white') 
    colors[0, :] = '#f0f0f0'
    colors[:, 0] = '#f0f0f0'

    for ind in areaInd:
        colors[ind[0], ind[1] + 1] = 'green'

    cell_height = 0.2
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
            table[(i, j)].set_height(0.3)

    outputPath = os.path.join(iterDir, f"T_{iterName}.png")
    plt.savefig(outputPath, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    return


def graphicManager(x, y, areaInd, iterDir, iterName):
    os.makedirs(iterDir, exist_ok=True)
    
    xx, yy = np.meshgrid(x, y)

    mask = np.full((len(x), len(y)), False)
    for i,j in areaInd:
        mask[i][j] = True

    plt.figure(figsize=(8, 6))
    plt.grid(True, linestyle=':', alpha=0.3, color='gray')
    plt.scatter(xx, yy, s=500, c='lightblue', alpha=0.7, edgecolors='gray')
    plt.scatter(xx[mask], yy[mask], s=500, c='red', edgecolors='black')

    outputPath = os.path.join(iterDir, f"GR_{iterName}.png")
    plt.savefig(outputPath, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    return


def findOptArea(arr, threshold):
    area = []
    minInd = np.unravel_index(np.argmin(arr), arr.shape)
    minVal = arr[minInd[0], minInd[1]]
    mask = arr <= (minVal + threshold)
    inds = np.where(mask)
    area = list(zip(inds[0], inds[1]))
    return area, minVal, minInd


def visualisationResults(gamma, systemInfo, T1, T4, threshold=1, iterDir="D:\PyProjects\Experiments", iterName="testSave"):
    '''queque1 = [q[-2][0] if q != None else np.nan for q in systemInfo]
    queque1 = np.array(queque1).reshape((len(T4), len(T1)))

    queque2 = [q[-2][1] if q != None else np.nan for q in systemInfo]
    queque2 = np.array(queque2).reshape((len(T4), len(T1)))

    dfQ1 = pd.DataFrame(queque1, index=T4, columns=T1).fillna(-1)
    dfQ2 = pd.DataFrame(queque2, index=T4, columns=T1).fillna(-1)'''

    gammaWeight = []
    #gamma1 = []
    #gamma2 = []
    for g in gamma:
        if None in g:
            gammaWeight.append(np.nan)
            #gamma1.append(np.nan)
            #gamma2.append(np.nan)
        else:
            gammaWeight.append((Lambda[0] * g[0] + Lambda[1] * g[1]) / (Lambda[0] + Lambda[1]))
            #gamma1.append(g[0])
            #gamma2.append(g[1])

    gammaWeight = np.array(gammaWeight).reshape((len(T4), len(T1)))
    optAreaIndexes, minG, minGIndex = findOptArea(gammaWeight, threshold)
    #gamma1 = np.array(gamma1).reshape((len(T4), len(T1)))
    #gamma2 = np.array(gamma2).reshape((len(T4), len(T1)))
    
    #dfG1 = pd.DataFrame(gamma1, index=T4, columns=T1).fillna(-1)
    #dfG2 = pd.DataFrame(gamma2, index=T4, columns=T1).fillna(-1)
    dfGammaWeight = pd.DataFrame(gammaWeight, index=T4, columns=T1).fillna(-1)

    graphicManager(T4, T1, optAreaIndexes, iterDir, iterName)
    tableManager(dfGammaWeight, optAreaIndexes, iterDir, iterName)

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


def experiment(r, g, type, q, si, numberOfServiceStates, nSt, stateTime, maxSumTimeOfState, tBounds1, tBounds2,
               eps=1, timeStep=1, flagForS=True, threshold=1,
               lambdaBounds=[[0.1, 0.6], [0.1, 0.6]], lmstep=0.1):
    iterCount = 0
    lmArr1 = np.arange(lambdaBounds[0][0], lambdaBounds[0][1] + lmstep, lmstep)
    lmArr2 = np.arange(lambdaBounds[1][0], lambdaBounds[1][1] + lmstep, lmstep)
    lmArr1 = np.round(lmArr1, 2)
    lmArr2 = np.round(lmArr2, 2)
    dir = f"D:\\PyProjects\\Experiments\\r={r[0]}_g={g[0]}"
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
    '''
    textT = []
    for arr in resOptT:
        text = ' '.join([f'{x:.2f}' for x in arr])
        textT.append([text])
    textT = np.array(textT).reshape((len(lmArr2), len(lmArr1)))
    dfRes = pd.DataFrame(textT, index=lmArr2, columns=lmArr1)
    num_rows, num_cols = dfRes.shape
    cell_text = []
    for idx, row in dfRes.iterrows():
        formatted_row = [val for val in row]
        cell_text.append([idx] + formatted_row)
    col_labels = [''] + dfRes.columns.tolist()
    cell_height = 0.4
    cell_width = 2   
    fig_width = max(6, num_cols * cell_width)  
    fig_height = max(4, num_rows * cell_height) 
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    font = FontProperties(family='monospace', weight='bold')
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center', colWidths=[0.2] + [0.15]*num_cols)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for i in range(num_rows + 1):
        table[(i, 0)].set_height(0.3)
        for j in range(1, num_cols + 1):
            table[(i, j)].set_height(0.3)
    outputPath =  f"D:\\PyProjects\\Experiments\\r={r[0]}_g={g[0]}\T2_r={r[0]}_g={g[0]}.png"
    plt.savefig(outputPath, dpi=300, bbox_inches='tight', pad_inches=0.5)'''
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
Threshold = 1 # Для построение квазиоптимальной области
#testSys = system.System(Lambda, Type, R, G, Q, SI, NumberOfServiceStates, Nst, StateTime)
#print(testSys.processing())
#systemWork(testSys, FlagForS)
#visualisationDynamics(testSys.getSystemDynamics(), testSys.getTimeArray(), testSys.getSystemQChanges())
#gamma, grid, T1, T4, systemInfo = optimizer([Lambda, Type, R, G, Q, SI, NumberOfServiceStates, Nst], [30, 35], [30, 35], StateTime, FlagForS, StepTime, Eps, MaxSumTimeOfStates)
#visualisationResults(gamma, systemInfo, T1, T4)

experiment(R, G, Type, Q, SI, NumberOfServiceStates, Nst, StateTime, 
           MaxSumTimeOfStates, [20, 25], [20, 25], Eps, StepTime, FlagForS, Threshold, [[0.1, 0.2], [0.1, 0.2]], 0.1)

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
