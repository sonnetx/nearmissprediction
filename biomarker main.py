import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

"""### Data Preparation"""

import pandas as pd

# df = pd.read_csv('MD_stress_dataset_cleaned.csv')
df = pd.read_csv('MD_stress_dataset.csv')
df.shape
df.head()

df = df[['CortDheaRatio_Post', 'CortDheaRatio_Pre', 'Cortisol_Post', 'DHEA_Post', 'Cortisol_Pre', 'DHEA_Pre', 'misses_res']]
df.head()

df.dropna(inplace=True)

df.isna().sum()

x = df[['CortDheaRatio_Post', 'CortDheaRatio_Pre', 'Cortisol_Post', 'DHEA_Post', 'Cortisol_Pre', 'DHEA_Pre']]
y = df[['misses_res']]

X = x.to_numpy()
y = y.to_numpy()

from collections import Counter
from imblearn.over_sampling import SMOTE 
from numpy import where
import matplotlib.pyplot as pyplot
# transform the dataset
y = y.reshape((31))

print(Counter(y))
sm = SMOTE(random_state=654)
# sm = SMOTE()
X_res, y_res = sm.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))
# summarize the new class distribution
counter = Counter(y)
print(counter)
# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 0], label=str(label))
pyplot.show()
X = X_res

x = X_res
y = y_res

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(x)

x = scaler.transform(x)


from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3 , random_state = 128)

# import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix

"""### Models"""

models = []
models.append(('SVM', svm.SVC(class_weight="balanced")))
models.append(('LSVM', SVC(kernel='linear', class_weight='balanced')))
models.append(('NB', GaussianNB()))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'

for name, model in models:
  kfold = model_selection.KFold(n_splits=2)
  cv_results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  
  model.fit(x_train, y_train)
  y_train_pred = model.predict(x_train)
  y_test_pred = model.predict(x_test)
  train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
  test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)

  plt.grid()

  plt.plot(train_fpr, train_tpr, label=" AUC TRAIN ="+str(auc(train_fpr, train_tpr)))
  plt.plot(test_fpr, test_tpr, label=" AUC TEST ="+str(auc(test_fpr, test_tpr)))
  plt.plot([0,1],[0,1],'g--')
  plt.legend()
  plt.xlabel("True Positive Rate")
  plt.ylabel("False Positive Rate")
  plt.title("AUC(ROC curve) " + name)
  plt.grid(color='black', linestyle='-', linewidth=0.5)
  plt.show()
  y_train = y_train.reshape((len(y_train),))