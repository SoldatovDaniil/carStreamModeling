import modeling

#Клас моделирующий входной поток
class InputStreamModeling:

    def __init__(self, lam, type = 'poisson', r = 0, g = 0):
        self.r = r
        self.g = g
        self.lam = lam
        self.type = type

    def setIntensity(self, lam):
        self.lam = lam

    def setR(self, r):
        self.r = r
    
    def setG(self, g):
        self.g = g

    def inputStreamModeling(self, time):
        moments = []
        if self.type == 'poisson':
            moments = modeling.OneStepPoisson(self.lam, time)
        elif self.type == 'bartlet':
            moments = modeling.getBartMoments(self.r, self.g, self.lam, time)
        return moments
