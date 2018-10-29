import numpy as np, matplotlib.pyplot as plt
x = np.array([[0.8, 0.6, 1.4], [0.3, 1.0, 1.3], [-0.3, 1.0, 0.6], [-0.8, 0.6, -0.2], [-1.0, 0.0, -1.0], [-0.8, -0.6, -1.4], [-0.3, -1.0, -1.3], [0.3, -1.0, -0.6], [0.8, -0.6, 0.2], [1.0, 0.0, 1.0]])
z = (x - np.mean(x, axis = 0)) / np.std(x, axis = 0)
(v,p) = np.linalg.eigh(1/len(x) * np.dot(np.transpose(z), z))
plt.scatter(np.dot(z, p[:,-1]), np.dot(z, p[:,-2]))
plt.show()