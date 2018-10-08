# Shivam Vyas

import random
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

cancer_set = datasets.load_breast_cancer()

X_ori = cancer_set["data"]
Y_ori = cancer_set["target"].astype(np.float64)
num_examples = X_ori.shape[0]
num_features = X_ori.shape[1]

print("Number of examples data points available: " + str(num_examples))
print("Number of input variables: " + str(num_features))

for i in range(0, 5 * num_examples):
    rand = random.randrange(0, num_examples)
    rand2 = random.randrange(0, num_examples)
    temp_x = X_ori[rand, :].copy()
    temp_x2 = X_ori[rand2, :].copy()
    temp_y = Y_ori[rand].copy()
    temp_y2 = Y_ori[rand2].copy()
    X_ori[rand, :] = temp_x2
    X_ori[rand2, :] = temp_x
    Y_ori[rand] = temp_y2
    Y_ori[rand2] = temp_y

test_set = 50

X_test = X_ori[0:test_set, :].copy()
Y_test = Y_ori[0:test_set].copy()

X_train = X_ori[test_set:, :]
Y_train = Y_ori[test_set:]

svm_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="poly", degree=5, coef0=1, C=5))
))

svm_clf.fit(X_train,Y_train)

false_negative = 0
false_positive = 0

for i in range(0, test_set):
    ans = svm_clf.predict([X_test[i, :]])
    error = ""

    if Y_test[i] == 1 and ans[0] == 0:
        false_negative += 1
    if Y_test[i] == 1 and ans[0] == 0:
        false_positive += 1

print(str(false_positive) + " False Positive(s) and " + str(false_negative)
      + " False negative(s) " + str(test_set - false_positive - false_negative)
      + " correct diagnosis")
