#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#where 0=case 1=control(no breast cancer)


# In[62]:


from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
df = pd.read_csv('C:\\Users\\Alamukii\\Desktop\\joined data.csv')
df['marital status'] == 'married'
df['married'] = df['marital status'] == 'married'
df['family history'] == 'yes'
df['yes family history'] = df['family history'] == 'yes'
x = df[['age','married','yes family history']].values
y = df['cancer'].values
x_train,x_test,y_train,y_test = train_test_split(x,y, random_state=22)
feature_names = ['age','married','yes family history']
target_names = ['cancer']
dt = tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, max_leaf_nodes=10)
dt.fit(x, y)
plt.figure(figsize = (30,15))
tree.plot_tree(dt,
              feature_names = feature_names,
              filled = True,
              rounded = True,
              fontsize = 25,);
plt.savefig('Breast Cancer Decision Tree.png')


# In[14]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
print('accuracy:', model.score (x_test, y_test))
ypred = model.predict(x_test)

print('precision:', precision_score(y_test, ypred))
print('recall:', recall_score(y_test, ypred))
print('f1 score:', f1_score(y_test, ypred))
print('accuracy:', accuracy_score(y_test,ypred))

kf = KFold(n_splits=5, shuffle=True)
for criterion in['gini', 'entropy']:
    print('Decision Tree - {}'.format(criterion))
accuracy =[]
precision = []
recall = []
for train_index, test_index in kf.split (x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    dt = DecisionTreeClassifier(criterion = criterion)
    dt.fit(x_train,y_train)
    ypred = dt.predict(x_test)
    accuracy.append(accuracy_score(y_test,ypred))
    precision.append(precision_score(y_test,ypred))
    recall.append(recall_score(y_test,ypred)) 
    print('accuracy:',np.mean(accufrom sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
print('accuracy:', model.score (x_test, y_test))
ypred = model.predict(x_test)

print('precision:', precision_score(y_test, ypred))
print('recall:', recall_score(y_test, ypred))
print('f1 score:', f1_score(y_test, ypred))
print('accuracy:', accuracy_score(y_test,ypred))

kf = KFold(n_splits=5, shuffle=True)
for criterion in['gini', 'entropy']:
    print('Decision Tree - {}'.format(criterion))
accuracy =[]
precision = []
recall = []
for train_index, test_index in kf.split (x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    dt = DecisionTreeClassifier(criterion = criterion)
    dt.fit(x_train,y_train)
    ypred = dt.predict(x_test)
    accuracy.append(accuracy_score(y_test,ypred))
    precision.append(precision_score(y_test,ypred))
    recall.append(recall_score(y_test,ypred)) 
    print('accuracy:',np.mean(accuracy))
    print('precision:',np.mean(precision))
    print('recall:',np.mean(recall))racy))
    print('precision:',np.mean(precision))
    print('recall:',np.mean(recall))


# In[54]:


from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
df = pd.read_csv('C:\\Users\\Alamukii\\Desktop\\joined data.csv')
df['marital status'] == 'married'
df['married'] = df['marital status'] == 'married'
df['family history'] == 'yes'
df['yes family history'] = df['family history'] == 'yes'
x = df[['age','married','yes family history']].values
y = df['cancer'].values
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
print('accuracy:', model.score (x_test, y_test))
ypred = model.predict(x_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
print('accuracy:', model.score (x_test, y_test))
ypred = model.predict(x_test)

print('precision:', precision_score(y_test, ypred))
print('recall:', recall_score(y_test, ypred))
print('f1 score:', f1_score(y_test, ypred))
print('accuracy:', accuracy_score(y_test,ypred))


x_train,x_test,y_train,y_test = train_test_split(x,y, random_state=22)
feature_names = ['age','married','yes family history']
target_names = ['cancer']
dt = tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, max_leaf_nodes=10)
dt.fit(x_train, y_train)
plt.figure(figsize = (30,15))
tree.plot_tree(dt,
              feature_names = feature_names,
              filled = True,
              rounded = True,
              fontsize = 25);
plt.savefig('Breast Cancer Decision Tree.png')


# In[ ]:




