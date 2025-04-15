import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Поток Пуассона
# Функция моделированния Пуассоновской случайной велечины
def PoissonRandomVariable(lam, t):
    p = random.random() # Датчик псевдослучайных чисел
    p *= np.exp(lam * t) # Выносим умноженеи на экспоненту
    S = 1 # "Вероятность" первого значения с. в.
    F = 0 # Граница интервала
    i = 0 # Текущее значение с.в.
    F += S
    while(p > F):
        i += 1
        S *= (lam * t) / i
        F += S
    N = i
    return(N)

# Получение моментов поступления требований в систему(равномерное распределение времени между заявками)
def UniformTimeMoments(t, appCount):
    moments = []
    for i in range(appCount):
        time = random.random() * t
        moments.append(time)
    moments.sort()
    return moments

# Моделирование потока Пуассона на одном промежутке
def OneStepPoisson(lT, t):
    aCounter = PoissonRandomVariable(lT, t)
    aMoments = UniformTimeMoments(t, aCounter)
    return aMoments

# Моделирование потока Пуассона
def ModelPoisson(lamTheor, time):
    appMoments = []
    lamStat = 0
    modelingCount = 0
    while abs(lamTheor - lamStat) > 0.1 * lamTheor:
        tmpMoments = OneStepPoisson(lamTheor, time)
        appMoments += [i + (modelingCount * time) for i in tmpMoments]
        modelingCount += 1
        lamStat = len(appMoments) / (modelingCount * time)
    return appMoments, lamStat, modelingCount

# Поиск минимального промежутка времени между прибытием  машин
def FindMinTimeGap(moments):
    if len(moments) == 0:
        return 0
    min_ = moments[-1]
    for i in range(len(moments) - 1):
        if moments[i + 1] - moments[i] < min_:
            min_ = moments[i + 1] - moments[i]
    return min_

# Визуализация моментов поступлений
def VisualisationPoisson(moments, lamStat = 0, modelingCount = 0):
    print(f"Количетсво машин: {len(moments)} \
          \nПромежутки моделирования: {modelingCount} \
          \nЛямбда статистическая: {lamStat} \
          \nМиниимальное расстояние между заявками {FindMinTimeGap(moments)}")
    yy = [0] * len(moments)
    plt.figure(figsize=(7 * modelingCount, 1))
    plt.scatter(moments, yy)
    plt.show()

# Поток Бартлетта
# Получение быстрых машин в пачке
def BartVariable(r, g):
    p = random.random() # Датчик псевдослучайных чисел
    S = 1 - r # "Вероятность" первого значения с. в.
    F = 0 # Граница интервала
    i = 0 # Текущее значение с.в.
    F += S
    while(p > F):
        i += 1
        if i == 1: 
            S = r * (1 - g)
            F += S
        else:
            S *= g
            F += S
    N = i
    return(N)

# Получение количества быстрых машин для каждой пачки
def FastCarInPacks(r, g, packs):
    fastPacks = []
    for i in range(len(packs)):
        fastPacks.append(BartVariable(r, g))
    return fastPacks

# Получение среднего числа машин в пачке
def FindMStat(packCount, fastOnPack):
    summ = 0
    for i in range(packCount):
        summ += fastOnPack[i] + 1
    return summ / packCount

# Получение r статистического
def FindRStat(packCount, fastOnPack):
    summ = 0
    for i in range(packCount):
        if fastOnPack[i] >= 1:
            summ += 1
    return summ / packCount

# Один промежуток моделирования потока Бартлета
def OneStepBart(r, g, lam, t, C = 2):
    M = 1 + r / (1 - g)
    lb = lam / M
    resMoments = []
    packs = OneStepPoisson(lb, t)
    fastCars = FastCarInPacks(r, g, packs)
    if len(packs) == 0:
        return resMoments, packs, fastCars
    delta = FindMinTimeGap(packs) / (C + max(fastCars))
    for i in range(len(packs)):
        resMoments.append(packs[i])
        for j in range(fastCars[i]):
            resMoments.append(resMoments[-1] + delta)
    return resMoments, packs, fastCars

# Получение моментов поступления машин
def getBartMoments(r, g, lam, t, C = 2):
    res, _, _  = OneStepBart(r, g, lam, t, C)
    return res

# Получение финального потока
def BartModel(r, g, lam, time, C = 2):
    lamStat = 0
    lamBStat = 0
    modelingCount = 0
    resPackCount = 0
    resPacksMoments = []
    resFastOnPacks = []
    resMoments = []
    while abs(lam - lamStat) > 0.1 * lam:
        tmpMoments, tmpPacks, tmpFastCars = OneStepBart(r, g, lam, time, C)
        resMoments += [i + (modelingCount * time) for i in tmpMoments]
        resPacksMoments += [i + (modelingCount * time) for i in tmpPacks]
        resFastOnPacks += tmpFastCars
        resPackCount += tmpPacks
        modelingCount += 1
        lamStat = len(resMoments) / (time * modelingCount)
    return resMoments, resPacksMoments, resFastOnPacks, lamStat, modelingCount


def VisualisationBartlet(moments, fc, lamStat = 0, modelingCount = 0):
    print(f"Количетсво машин: {len(moments)} \
          \nКоличество быстрых машин в пачке {fc} \
          \nПромежутки моделирования: {modelingCount} \
          \nЛямбда статистическая: {lamStat} \
          \nМиниимальное расстояние между заявками {FindMinTimeGap(moments)}")
    yy = [0] * len(moments)
    plt.figure(figsize=(7 * modelingCount, 1))
    plt.scatter(moments, yy)
    plt.show()


"""T = 50
Lmd = 0.3
R = 0.5
G = 0.5
moments, packMoments, fastCars, lst, mc = BartModel(R, G, Lmd, T)
VisualisationPoisson(packMoments)
VisualisationBartlet(moments, fastCars, lst, mc)"""


"""def Queue(r, g, lam, qtime):
    appMoments, p, fc = OneStepBart(r, g, lam, qtime)
    print(len(appMoments))
    queue = [qtime - i for i in appMoments]
    return queue

print(Queue(0.5, 0.5, 0.3, 35))"""
