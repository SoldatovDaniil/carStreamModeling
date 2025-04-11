import streamModeling

# Класс реализующий работу одного потока
class streamService:
    
    def __init__(self, inputStream : streamModeling.InputStreamModeling, queue : list, serviceIntensity : list, servicedCount = 0, estMean = 0, estVar = 0):
        self.inputStream = inputStream # Входной поток
        self.q = queue # Очередь - времена ожидания машин
        self.n = servicedCount # Количество обслуженных машин
        self.inputCarsCount = 0 # Количество поступивших машин
        self.intensity1, self.intensity2 = serviceIntensity # Интенсивность обслуживания зелёный свет и мигающий зелёный
        self.g = estMean # Оценка среднего времени задержки машины в системе
        self.s = estVar # Оценка дисперсии среднего времени задержки машины в системе

    def setInputStream(self, newStream): # Установка входного потока
        self.inputStream = newStream

    def setServiceIntensity(self, newInt): # Смена интенсивности осблуживания
        self.intensity = newInt

    def addWaitTime(self, time): # Добавление времени ожидания машинам
        for i in range(len(self.q)):
            self.q[i] += time

    def addNewApp(self, time, moments): # Добавление новых заявок в очередь
        for i in range(len(moments)):
            self.q.append(time - moments[i])

    def noServicePhase(self, time): # Фаза без обслуживания
        moments = self.inputStream.inputStreamModeling(time)
        self.inputCarsCount += len(moments)
        self.addWaitTime(time)
        self.addNewApp(time, moments)

    def firstCase(self, time, moments, maxServedCount): # Случай первый - очередь больше, чем можно обслужить
        for i in range(maxServedCount):
            self.g = (self.g * self.n + self.q[i]) / (self.n + 1)
            self.s = (self.s * self.n + self.q[i] ** 2) / (self.n + 1)
            self.n += 1
        if maxServedCount == len(self.q):
            self.q = []
        else:
            self.q = self.q[maxServedCount:]
        self.addWaitTime(time)
        self.addNewApp(time, moments)

    def secondCase(self, time, moments, maxServedCount, intensity): # Случай второй - обслужаться все из очереди и вновь пришедшие
        for i in range(len(self.q)):
            qi = self.q[i]
            self.g = (self.g * self.n + qi) / (self.n + 1)
            self.s = (self.s * self.n + qi ** 2) / (self.n + 1)
            self.n += 1
        self.q = []
        l = maxServedCount * intensity
        for i in range(len(moments)):
            if moments[i] >= l and time - moments[i] > intensity:
                l = moments[i] + intensity
                qi = intensity
                self.g = (self.g * self.n + qi) / (self.n + 1)
                self.s = (self.s * self.n + qi ** 2) / (self.n + 1)
                self.n += 1
            elif time - moments[i] > intensity:
                qi = l - moments[i] + intensity
                l += intensity
                self.g = (self.g * self.n + qi) / (self.n + 1)
                self.s = (self.s * self.n + qi ** 2) / (self.n + 1)
                self.n += 1
            else:
                qi = intensity
                self.g = (self.g * self.n + qi) / (self.n + 1)
                self.s = (self.s * self.n + qi ** 2) / (self.n + 1)
                self.n += 1
    
    def thirdCase(self, time, moments, maxServedCount, intensity): # Случай третий - обслужаться все из очереди, но не все пришедшие 
        qN = len(self.q)
        for i in range(qN):
            qi = self.q[i]
            self.g = (self.g * self.n + qi) / (self.n + 1)
            self.s = (self.s * self.n + qi ** 2) / (self.n + 1)
            self.n += 1
        self.q = []
        l = maxServedCount * intensity
        for i in range(maxServedCount - qN):
            if moments[i] >= l and time - moments[i] > intensity:
                l = moments[i] + intensity
                qi = intensity
                self.g = (self.g * self.n + qi) / (self.n + 1)
                self.s = (self.s * self.n + qi ** 2) / (self.n + 1)
                self.n += 1
            elif time - moments[i] > intensity:
                qi = l - moments[i] + intensity
                l += intensity
                self.g = (self.g * self.n + qi) / (self.n + 1)
                self.s = (self.s * self.n + qi ** 2) / (self.n + 1)
                self.n += 1
            else:
                qi = intensity
                self.g = (self.g * self.n + qi) / (self.n + 1)
                self.s = (self.s * self.n + qi ** 2) / (self.n + 1)
                self.n += 1
        self.addNewApp(time, moments[(maxServedCount - qN):])
        
    def servicePhase(self, time, numServicePhase): # Фаза обслуживания
        if numServicePhase == 1:
            intensity = self.intensity1
        else:
            intensity = self.intensity2
        maxServedCount = time // intensity
        moments = self.inputStream.inputStreamModeling(time)
        self.inputCarsCount += len(moments)
        if len(self.q) >= maxServedCount:
            self.firstCase(time, moments, maxServedCount)
        elif len(moments) <= maxServedCount - len(self.q):
            self.secondCase(time, moments, maxServedCount, intensity)
        elif len(moments) > maxServedCount - len(self.q):
            self.thirdCase(time, moments, maxServedCount, intensity)
    
    def getRes(self): # Получение результатов
        resS = self.s - self.g ** 2
        return self.inputCarsCount, self.n, self.g, resS

    def getQ(self): # Получение очереди
        return self.q
    
    def printQueue(self): # Вывод очереди
        print(self.q)

# Класс реализующий работу системы
class system:
    def __init__(self, Lam : list, Type : list, R : list, G : list,
                 Queues : list, ServiceIntensity : list,
                 Nstop, T : list, Qmax = 1000, EPS = 1): 
        self.is1 =  streamModeling.InputStreamModeling(Lam[0], Type[0], R[0], G[0]) # Входной поток 1
        self.is2 =  streamModeling.InputStreamModeling(Lam[1], Type[1], R[1], G[1]) # Входной поток 1
        self.s1 =  streamService(self.is1, Queues[0], ServiceIntensity[0]) # Поток 1
        self.s2 =  streamService(self.is2, Queues[1], ServiceIntensity[1]) # Поток 2
        self.Nstop = Nstop # Максимальное количество обслуженных машин
        self.T = T # Массив времён с длительностями состояний 
        self.EPS = EPS # Параметр для расчёта оценок
        self.Qmax = Qmax # Максимальное значение машин в очереди
    
    def proccessing(self): # Моделирование работы системы
        totalN1 = 0
        totalN2 = 0
        cycleCount = 0
        Flag = True
        while totalN1 < self.Nstop or totalN2 < self.Nstop: 
            for i in range(len(self.T)):
                if len(self.s1.getQ()) > self.Qmax or len(self.s2.getQ()) > self.Qmax:
                    n1, g1, s1 = self.s1.getRes()
                    n2, g2, s2 = self.s2.getRes()
                    totalN1 += n1
                    totalN2 += n2
                    return g1, g2, s1, s2, cycleCount, totalN1, totalN2
                if i == 0:
                    self.s1.servicePhase(self.T[i], 1)
                    self.s2.noServicePhase(self.T[i])
                elif i == 1:
                    self.s1.servicePhase(self.T[i], 2)
                    self.s2.noServicePhase(self.T[i])
                elif i == 2 or i == 5:
                    self.s1.noServicePhase(self.T[i])
                    self.s2.noServicePhase(self.T[i])
                elif i == 3:
                    self.s1.noServicePhase(self.T[i])
                    self.s2.servicePhase(self.T[i], 1)
                elif i == 4:
                    self.s1.noServicePhase(self.T[i])
                    self.s2.servicePhase(self.T[i], 2)
                
            inCars1, n1, g1, s1 = self.s1.getRes()
            inCars2, n2, g2, s2 = self.s2.getRes()
            totalN1 = n1
            totalN2 = n2
            cycleCount += 1
            
        return g1, g2, s1, s2, cycleCount, totalN1, totalN2, inCars1, inCars2

Lambda = [0.6, 0.6]
Type = ['bartlet', 'bartlet']
R = [0.5, 0.5]
G = [0.4, 0.4]
Q = [[], []]
SI = [[1, 2], [1, 2]]
Nst = 10000
StateTime= [15, 3, 3, 15, 3, 3]
testSys = system(Lambda, Type, R, G, Q, SI, Nst, StateTime)
print(testSys.proccessing())