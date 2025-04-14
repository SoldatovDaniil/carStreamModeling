import system


Lambda = [0.6, 0.6]
Type = ['bartlet', 'bartlet']
R = [0.5, 0.5]
G = [0.4, 0.4]
Q = [[], []]
SI = [[1, 2], [1, 2]]
Nst = 10000
StateTime= [15, 3, 3, 15, 3, 3]
testSys = system.System(Lambda, Type, R, G, Q, SI, Nst, StateTime)
print(testSys.proccessing())