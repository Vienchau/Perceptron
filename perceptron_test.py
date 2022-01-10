import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import datasets #datasets là bộ dữ liệu được chuẩn hóa trong thư viện 
import matplotlib.pyplot as plt

from perceptron  import  Perceptron

#n_sample: total point  equally amon cluster, centers: số trung tâm các điểm, n_features: số thuộc tính, cluster_std: càng lớn phạm vi sai (xa trung tâm) càng tăn
X, y = datasets.make_blobs(n_samples = 200, n_features = 2, centers = 2, cluster_std = 1.1, random_state = None )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=123)

p = Perceptron(learning_rate = 0.01, n_iters = 1000)
p.fit(X_train, y_train)
predictions = p.predict(X_test)


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.scatter(X_train[:,0],X_train[:,1], marker = 'o', c=y_train)

x0_1 = np.amin(X_train[:,0])
x0_2= np.amax(X_train[:,0])

x1_1 = (-p.weights[0] *x0_1 - p.bias)/p.weights[1]
x1_2 = (-p.weights[0] *x0_2 - p.bias)/p.weights[1]

ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')

ymin = np.amin(X_train[:,1])
ymax = np.amax(X_train[:,1])
ax.set_ylim([ymin-3, ymax +3])

plt.show()