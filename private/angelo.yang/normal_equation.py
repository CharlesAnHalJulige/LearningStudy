import numpy as np
import matplotlib.pyplot as plt

# size, number of bedrooms, number of floors, age of home, price(value to predict)
data = np.array([
    [2104, 5, 1, 45, 460],
    [1416, 5, 2, 40, 232],
    [1534, 3, 2, 30, 315],
    [852, 2, 1, 36, 178]
])

# calculate optimal parameters by normal equation
m = len(data)
X = np.concatenate((np.ones((m, 1)), data[:, :-1]), axis=1)
y = np.array(data[:, -1]).T
parameters = np.linalg.solve(X.T.dot(X), X.T.dot(y))

# print
np.set_printoptions(precision=3)
print(parameters)
