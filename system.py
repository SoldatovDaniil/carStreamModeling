import streamModeling

# Класс реализующий работу одного потока
class StreamService:
    
    def __init__(self, inputStream : streamModeling.InputStreamModeling, queue : list, 
                 serviceIntensity : list, numbersOfServiceStates : list, 
                 servicedCount = 0, estMean = 0, estVar = 0):
        self.inputStream = inputStream # Входной поток
        self.q = queue # Очередь - времена ожидания машин, время их поступления (мемент поступления, время задержки)
        self.queueChanges = [] # Массив для отслеживания изменений очередей (момент времени, длина очереди)
        self.n = servicedCount # Количество обслуженных машин
        self.inputCarsCount = 0 # Количество поступивших машин
        self.numbersOfServiceStates = numbersOfServiceStates # Номера фаз обслуживания
        self.intensity1, self.intensity2 = serviceIntensity # Интенсивность обслуживания зелёный свет и мигающий зелёный
        self.g = estMean # Оценка среднего времени задержки машины в системе
        self.s = estVar # Оценка дисперсии среднего времени задержки машины в системе

    def setInputStream(self, newStream): # Установка входного потока
        self.inputStream = newStream

    def setServiceIntensity(self, newInt): # Смена интенсивности осблуживания
        self.intensity = newInt

    def addWaitTime(self, stateTime): # Добавление времени ожидания машинам
        for i in range(len(self.q)):
            self.q[i][1] += stateTime

    def searchIntensity(self, state): # Нахождение интенсивности в фазе обслуживания(для добавления машин в очередь)
        if state == self.numbersOfServiceStates[1]:
            return self.intensity2
        else:
            return self.intensity1
            
        
    def addNewApp(self, stateTime, moments, systemTime, sys): # Добавление новых заявок в очередь
        if len(self.q) == 0:
            self.q.append(systemTime + moments[0], stateTime - moments[0])
        else:
            self.q.append(systemTime + moments[0], (self.q[-1][1] + self.searchIntensity(sys.searchState())) - (systemTime + moments[0] - self.q[-1][0]))

        for i in range(1, len(moments)):
            self.q.append([moments[i], stateTime - moments[i]])

    def noServicePhase(self, time): # Фаза без обслуживания
        moments = self.inputStream.inputStreamModeling(time)
        self.inputCarsCount += len(moments)
        self.addWaitTime(time)
        self.addNewApp(time, moments)

    def firstCase(self, time, moments, maxServedCount): # Случай первый - очередь больше, чем можно обслужить
        for i in range(maxServedCount):
            self.g = (self.g * self.n + self.q[i][1]) / (self.n + 1)
            self.s = (self.s * self.n + self.q[i][1] ** 2) / (self.n + 1)
            self.n += 1
            self.queueChanges.append([self.q[i][0] + self.q[i][1], len(self.q) - i - 1])
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
        
    def servicePhase(self, stateTime, numServicePhase): # Фаза обслуживания
        if numServicePhase == 1:
            intensity = self.intensity1
        else:
            intensity = self.intensity2
        maxServedCount = stateTime // intensity
        moments = self.inputStream.inputStreamModeling(stateTime)
        self.inputCarsCount += len(moments)
        if len(self.q) >= maxServedCount:
            self.firstCase(stateTime, moments, maxServedCount)
        elif len(moments) <= maxServedCount - len(self.q):
            self.secondCase(stateTime, moments, maxServedCount, intensity)
        elif len(moments) > maxServedCount - len(self.q):
            self.thirdCase(stateTime, moments, maxServedCount, intensity)
    
    def getRes(self): # Получение результатов
        resS = self.s - self.g ** 2
        return self.inputCarsCount, self.n, self.g, resS

    def getQ(self): # Получение очереди
        return self.q
    
    def getLenQ(self): # Получение длины очереди
        return len(self.q) 
    
    def printQueue(self): # Вывод очереди
        print(self.q)

# Класс реализующий работу системы
class System:
    def __init__(self, Lam : list, Type : list, R : list, G : list,
                 Queues : list, ServiceIntensity : list, NumbersOfServiceStates : list,
                 Nstop, T : list, Qmax = 1000): 
        self.is1 =  streamModeling.InputStreamModeling(Lam[0], Type[0], R[0], G[0]) # Входной поток 1
        self.is2 =  streamModeling.InputStreamModeling(Lam[1], Type[1], R[1], G[1]) # Входной поток 1
        self.s1 =  StreamService(self.is1, Queues[0], ServiceIntensity[0], NumbersOfServiceStates[0]) # Поток 1
        self.s2 =  StreamService(self.is2, Queues[1], ServiceIntensity[1], NumbersOfServiceStates[1]) # Поток 2
        self.Nstop = Nstop # Максимальное количество обслуженных машин
        self.T = T # Массив времён с длительностями состояний 
        self.Qmax = Qmax # Максимальное значение машин в очереди
    
    def setNstop(self, newNstop):
        self.Nstop = newNstop
    
    def getNstop(self):
        return self.Nstop
    
    def getQ(self):
        return len(self.s1.getQ()), len(self.s2.getQ())
    
    def searchState(self, time):
        time %= sum(self.T)
        for i in range(1, len(self.T)):
            if time >= sum(self.T[:i]) and time < sum(self.T[:i + 1]):
                return i + 1
            else:
                return 1
        
    def processing(self): # Моделирование работы системы
        totalN1 = 0
        totalN2 = 0
        cycleCount = 0
        flag = True
        while totalN1 < self.Nstop or totalN2 < self.Nstop: 
            for i in range(len(self.T)):
                if len(self.s1.getQ()) > self.Qmax or len(self.s2.getQ()) > self.Qmax: # Отсутствие стационара
                    inCars1, n1, g1, s1 = self.s1.getRes()
                    inCars2, n2, g2, s2 = self.s2.getRes()
                    totalN1 = n1
                    totalN2 = n2
                    flag = False
                    return g1, g2, s1, s2, cycleCount, totalN1, totalN2, inCars1, inCars2, self.getQ(), flag
                
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
            
        return g1, g2, s1, s2, cycleCount, totalN1, totalN2, inCars1, inCars2, self.getQ(), flag

