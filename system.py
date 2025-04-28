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
        self.intensity1, self.intensity2 = serviceIntensity # Интенсивность обслуживания зелёный свет и мигающий зелёный(время на проезд 1 машины т.е. 1/mu)
        self.g = estMean # Оценка среднего времени задержки машины в системе
        self.s = estVar # Оценка дисперсии среднего времени задержки машины в системе

    def setInputStream(self, newStream): # Установка входного потока
        self.inputStream = newStream

    def setServiceIntensity(self, newInt): # Смена интенсивности осблуживания
        self.intensity = newInt

    def addWaitTime(self, stateTime): # Добавление времени ожидания машинам
        for i in range(len(self.q)):
            self.q[i][1] += stateTime

    def searchIntensity(self, state, timeGap): # Нахождение интенсивности в фазе обслуживания(для добавления машин в очередь)
        if state == self.numbersOfServiceStates[1] and timeGap >= self.intensity2:
            return self.intensity2
        elif state == self.numbersOfServiceStates[0] and timeGap < self.intensity1:
            return self.intensity2
        else:
            return self.intensity1
            
    def addNewApp(self, stateTime, moments, systemTime, sys): # Добавление новых заявок в очередь
        if len(self.q) == 0:
            self.q.append([systemTime + moments[0], stateTime - moments[0]])
            self.queueChanges.append([self.q[-1][0], len(self.q)])
        else:
            self.q.append([systemTime + moments[0], (self.q[-1][0] + self.q[-1][1] + self.searchIntensity(*sys.searchState(self.q[-1][0] + self.q[-1][1]))) - (systemTime + moments[0])])
            self.queueChanges.append([self.q[-1][0], len(self.q)])
        for i in range(1, len(moments)):
            self.q.append([systemTime + moments[i], (self.q[-1][0] + self.q[-1][1] + self.searchIntensity(*sys.searchState(self.q[-1][0] + self.q[-1][1]))) - (systemTime + moments[i])])
            self.queueChanges.append([self.q[-1][0], len(self.q)])

    def noServicePhase(self, stateTime, systemTime, sys): # Фаза без обслуживания
        moments = self.inputStream.inputStreamModeling(stateTime)
        self.inputCarsCount += len(moments)
        self.addWaitTime(stateTime)
        if len(moments) > 0:
            self.addNewApp(stateTime, moments, systemTime, sys)

    def firstCase(self, time, moments, maxServedCount, systemTime, sys): # Случай первый - очередь больше, чем можно обслужить
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
        if len(moments) > 0:
            self.addNewApp(time, moments, systemTime, sys)

    def secondCase(self, time, moments, maxServedCount, intensity, systemTime): # Случай второй - обслужаться все из очереди и все вновь пришедшие
        qN = len(self.q)
        for i in range(qN):
            qi = self.q[i][1]
            self.g = (self.g * self.n + qi) / (self.n + 1)
            self.s = (self.s * self.n + qi ** 2) / (self.n + 1)
            self.n += 1
            self.queueChanges.append([self.q[i][0] + self.q[i][1], len(self.q) - i - 1])
        l = qN * intensity
        self.q = []
        for i in range(len(moments)):
            if moments[i] >= l and time - moments[i] > intensity:
                l = moments[i] + intensity
                qi = 0
                self.g = (self.g * self.n + qi) / (self.n + 1)
                self.s = (self.s * self.n + qi ** 2) / (self.n + 1)
                self.n += 1
            elif moments[i] < l and time - moments[i] > intensity:
                qi = l - moments[i]
                l += intensity
                self.g = (self.g * self.n + qi) / (self.n + 1)
                self.s = (self.s * self.n + qi ** 2) / (self.n + 1)
                self.n += 1
            elif time - moments[i] <= intensity:
                qi = 0
                self.g = (self.g * self.n + qi) / (self.n + 1)
                self.s = (self.s * self.n + qi ** 2) / (self.n + 1)
                self.n += 1
    
    def thirdCase(self, time, moments, maxServedCount, intensity, systemTime, sys): # Случай третий - обслужаться все из очереди, но не все пришедшие 
        qN = len(self.q)
        for i in range(qN):
            qi = self.q[i][1]
            self.g = (self.g * self.n + qi) / (self.n + 1)
            self.s = (self.s * self.n + qi ** 2) / (self.n + 1)
            self.n += 1
            self.queueChanges.append([self.q[i][0] + self.q[i][1], len(self.q) - i - 1])
        l = qN * intensity
        self.q = []
        for i in range(maxServedCount - qN):
            if moments[i] >= l and time - moments[i] > intensity:
                l = moments[i] + intensity
                qi = 0
                self.g = (self.g * self.n + qi) / (self.n + 1)
                self.s = (self.s * self.n + qi ** 2) / (self.n + 1)
                self.n += 1
            elif moments[i] < l and time - moments[i] > intensity:
                qi = l - moments[i]
                l += intensity
                self.g = (self.g * self.n + qi) / (self.n + 1)
                self.s = (self.s * self.n + qi ** 2) / (self.n + 1)
                self.n += 1
            elif time - moments[i] <= intensity:
                qi = 0
                self.g = (self.g * self.n + qi) / (self.n + 1)
                self.s = (self.s * self.n + qi ** 2) / (self.n + 1)
                self.n += 1
        if len(moments) > 0:
            self.addNewApp(time, moments[(maxServedCount - qN):], systemTime, sys)
        
    def servicePhase(self, stateTime, numServicePhase, systemTime, sys): # Фаза обслуживания
        if numServicePhase == 1:
            intensity = self.intensity1
        else:
            intensity = self.intensity2
        maxServedCount = int(stateTime // intensity)
        moments = self.inputStream.inputStreamModeling(stateTime)
        self.inputCarsCount += len(moments)
        if len(self.q) >= maxServedCount:
            self.firstCase(stateTime, moments, maxServedCount, systemTime, sys)
        elif len(moments) <= maxServedCount - len(self.q):
            self.secondCase(stateTime, moments, maxServedCount, intensity, systemTime)
        elif len(moments) > maxServedCount - len(self.q):
            self.thirdCase(stateTime, moments, maxServedCount, intensity, systemTime, sys)
    
    def getRes(self): # Получение результатов
        resS = self.s - self.g ** 2
        return self.inputCarsCount, self.n, self.g, resS

    def getQ(self): # Получение очереди
        return self.q
    
    def getLenQ(self): # Получение длины очереди
        return len(self.q) 
    
    def getQChanges(self): # Получение изменений очереди
        return self.queueChanges
    
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
    
    def getSystemQChanges(self):
        return self.s1.getQChanges(), self.s2.getQChanges()
    
    def searchState(self, time):
        time %= sum(self.T)
        for i in range(1, len(self.T)):
            if time >= sum(self.T[:i]) and time < sum(self.T[:i+1]):
                return i + 1, sum(self.T[:i+1]) - time
            else:
                return 1, self.T[0] - time
        
    def processing(self): # Моделирование работы системы
        totalN1 = 0
        totalN2 = 0
        cycleCount = 0
        systemTime = 0 # Время начала текущей фазы обслуживания
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
                    self.s1.servicePhase(self.T[i], 1, systemTime, self)
                    self.s2.noServicePhase(self.T[i], systemTime, self)
                elif i == 1:
                    self.s1.servicePhase(self.T[i], 2, systemTime, self)
                    self.s2.noServicePhase(self.T[i], systemTime, self)
                elif i == 2 or i == 5:
                    self.s1.noServicePhase(self.T[i], systemTime, self)
                    self.s2.noServicePhase(self.T[i], systemTime, self)
                elif i == 3:
                    self.s1.noServicePhase(self.T[i], systemTime, self)
                    self.s2.servicePhase(self.T[i], 1, systemTime, self)
                elif i == 4:
                    self.s1.noServicePhase(self.T[i], systemTime, self)
                    self.s2.servicePhase(self.T[i], 2, systemTime, self)
                
                systemTime += self.T[i]
                
            inCars1, n1, g1, s1 = self.s1.getRes()
            inCars2, n2, g2, s2 = self.s2.getRes()
            totalN1 = n1
            totalN2 = n2
            cycleCount += 1
            
        return g1, g2, s1, s2, cycleCount, totalN1, totalN2, inCars1, inCars2, self.getQ(), flag



