import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import datasets #datasets là bộ dữ liệu được chuẩn hóa trong thư viện 
import matplotlib.pyplot as plt

from perceptron  import  Perceptron

#n_sample: total point  equally amon cluster, centers: số trung tâm các điểm, n_features: số thuộc tính, cluster_std: càng lớn phạm vi sai (xa trung tâm) càng tăn
X, y = datasets.make_blobs(n_samples = 100, n_features = 2, centers = 2, cluster_std = 1.1, random_state = None )

#lấy tỉ lệ 20% của tập dữ liệu để test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=123)

p = Perceptron(learning_rate = 0.01, n_iters = 1000)
#train
p.fit(X_train, y_train)
#danh gia predict

def test(y_true, y_pred):
	#so sánh y_true và y_pred giống nhau, giả sử acc = 0.8 vậy cứ 10 point test thì có 8 point giống
	acc =  np.sum(y_true == y_pred)/ len(y_true)
	return acc

predictions = p.predict(X_test)
accu = (test(y_test, predictions) * 100)

fig = plt.figure()
#add_subplot(1 hàng biểu đồ, 1 cột biểu đồ, stt 1)
ax = fig.add_subplot(1,1,1)
#X_train[:,0] in numpy: ALL row in collum 0
plt.scatter(X_train[:,0],X_train[:,1], marker = 'o', c=y_train)
ax.set_xlabel('X1 FEATURE')
ax.set_ylabel('X2 FEATURE')
ax.set_title('PERCEPTRON TEST')
plt.legend(title = 'Accuracy: %f percent'%accu)

x0_1 = np.amin(X_train[:,0])
x0_2= np.amax(X_train[:,0])

x1_1 = (-p.weights[0] *x0_1 - p.bias)/p.weights[1]
x1_2 = (-p.weights[0] *x0_2 - p.bias)/p.weights[1]

ax.plot([x0_1, x0_2], [x1_1, x1_2]) 

#ymin = np.amin(X_train[:,1])
#ymax = np.amax(X_train[:,1])
#ax.set_ylim([ymin-3, ymax +3])

plt.show()