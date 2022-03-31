#!/usr/bin/env python
# coding: utf-8

# ## creating a logistic regression model
# where 0=case 1=control(no breast cancer)
# 
# 

# In[51]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
df = pd.read_csv('C:\\Users\\Alamukii\\Desktop\\joined data.csv')
df['marital status'] == 'married'
df['married'] = df['marital status'] == 'married'
df['family history'] == 'yes'
df['yes family history'] = df['family history'] == 'yes'
x = df[['age','married','yes family history']].values
y = df['cancer'].values

model = LogisticRegression()
model.fit(x , y)
y_pred = model.predict(x)
print(model.predict([[22,  False, False]]))
model.score(x, y)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print('accuracy:', accuracy_score(y, y_pred))
print('precision:', precision_score(y, y_pred))
print('recall:', recall_score(y, y_pred))
print('f1 score:', f1_score(y, y_pred))


# confusion matrix
# 

# In[91]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y, y_pred))


# In[42]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y)

print('whole dataset:', x.shape, y.shape)
print('training set:', x_train.shape, y_train.shape)
print('test set:', x_test.shape, y_test.shape)


# In[43]:


model = LogisticRegression()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))


# In[63]:


import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_recall_fscore_support

sensitivity_score = recall_score
def specificity_score(y_true, y_pred):p,r,f,s = precision_recall_fscore_support(y_true,y_pred)
 
df = pd.read_csv('C:\\Users\\Alamukii\\Desktop\\joined data.csv')
df['marital status'] == 'married'
df['married'] = df['marital status'] == 'married'
df['family history'] == 'yes'
df['yes family history'] = df['family history'] == 'yes'
x = df[['age','married','yes family history']].values
y = df['cancer'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 5)

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('sensitivity:', sensitivity_score(y_test, y_pred))
print ('specificity:',specificity_score(y_test, y_pred))
model.predict_proba(x_test)
y_pred = model.predict_proba(x_test)[:,1]>0.75
print(y_pred)
print('precision:', precision_score(y_test, y_pred))
print('recall:',recall_score(y_test,y_pred))


# In[62]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred_proba = model.predict_proba(x_test)
fpr, tpr, thresholds = roc_curve(y_test,y_pred_proba[:,1])
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1], linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('FALSE POSITIVE RATE (1-specificity)')
plt.ylabel('TRUE POSITIVE RATE(sensitivity)')
plt.show()


# In[52]:


from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import pandas as pd 
df = pd.read_csv('C:\\Users\\Alamukii\\Desktop\\joined data.csv')
df['marital status'] == 'married'
df['married'] = df['marital status'] == 'married'
df['family history'] == 'yes'
df['yes family history'] = df['family history'] == 'yes'
x = df[['age','married','yes family history']].values
y = df['cancer'].values
kf = KFold(n_splits=5, shuffle=True)
splits = list(kf.split(x))
train_indices, test_indices = splits[0]

x_train = x[train_indices]
x_test = x[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

model = LogisticRegression()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
import numpy as np
scores = []
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = LogisticRegression()
    model.fit(x_train, y_train) 
    scores.append(model.score(x_test, y_test))
    print(scores)
    print(np.mean(scores))
    finalmodel = LogisticRegression()
finalmodel.fit(x, y)



# In[53]:


import numpy as np
scores = []
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = LogisticRegression()
    model.fit(x_train, y_train) 
    scores.append(model.score(x_test, y_test))
    print(scores)
    print(np.mean(scores))
    finalmodel = LogisticRegression()
finalmodel.fit(x, y)





# In[22]:


finalmodel = LogisticRegression()
finalmodel.fit(x, y)


# In[ ]:





# In[106]:


## JUST THE GENES

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd 
import numpy as np
df = pd.read_csv('C:\\Users\\Alamukii\\Desktop\\joined data.csv')
df['488G'] == 'G'
df['488GGENE'] = df['488G'] == 'G'
df['488A'] == 'A'
df['488AGENE'] = df['488A'] == 'A'
df['308A'] == 'A'
df['308AGENE'] = df['308A'] == 'A'
df['308G'] == 'G'
df['308GGENE'] = df['308G'] == 'G'
df['380G'] == 'G'
df['380GGENE'] = df['380G'] == 'G'
df['380A'] == 'A'
df['380AGENE'] = df['380A'] == 'A'
df['859C'] == 'C'
df['859CGENE'] = df['859C'] == 'C'
df['859T'] == 'T'
df['859TGENE'] = df['859T'] == 'T'
df['3W-A'] == 'A'
df['3W-AGENE'] = df['3W-A'] == 'A'
df['3M-G'] == 'G'
df['3M-GGENE'] = df['3M-G'] == 'G'
df['1032 C'] == 'C'
df['1032 CGENE'] = df['1032 C'] == 'C'
df['1032T'] == 'T'
df['1032TGENE'] = df['1032T'] == 'T'
x = df[['488GGENE','488AGENE','308AGENE','308GGENE','380GGENE','380AGENE','859CGENE','859TGENE','3W-AGENE','3M-GGENE','1032 CGENE','1032TGENE']].values
y = df['cancer'].values
model = LogisticRegression()
model.fit(x,y)
y_pred = model.predict(x)
model.score(x, y)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print('accuracy:', accuracy_score(y, y_pred))
print('precision:', precision_score(y, y_pred))
print('recall:', recall_score(y, y_pred))
print('f1 score:', f1_score(y, y_pred))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y, y_pred))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y)

print('whole dataset:', x.shape, y.shape)
print('training set:', x_train.shape, y_train.shape)
print('test set:', x_test.shape, y_test.shape)

model = LogisticRegression()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))



import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_recall_fscore_support

sensitivity_score = recall_score
def specificity_score(y_true, y_pred):p,r,f,s = precision_recall_fscore_support(y_true,y_pred)
 
df = pd.read_csv('C:\\Users\\Alamukii\\Desktop\\joined data.csv')
df['marital status'] == 'married'
df['married'] = df['marital status'] == 'married'
df['family history'] == 'yes'
df['yes family history'] = df['family history'] == 'yes'
x = df[['age','married','yes family history']].values
y = df['cancer'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 5)

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('sensitivity:', sensitivity_score(y_test, y_pred))
print ('specificity:',specificity_score(y_test, y_pred))
model.predict_proba(x_test)
y_pred = model.predict_proba(x_test)[:,1]>0.75
print(y_pred)
print('precision:', precision_score(y_test, y_pred))
print('recall:',recall_score(y_test,y_pred))


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred_proba = model.predict_proba(x_test)
fpr, tpr, thresholds = roc_curve(y_test,y_pred_proba[:,1])
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1], linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('FALSE POSITIVE RATE (1-specificity)')
plt.ylabel('TRUE POSITIVE RATE(sensitivity)')
plt.show()


# In[107]:


##  GENES AND OTHER FEATURES
 
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd 
import numpy as np
df = pd.read_csv('C:\\Users\\Alamukii\\Desktop\\joined data.csv')
df['marital status'] == 'married'
df['married'] = df['marital status'] == 'married'
df['family history'] == 'yes'
df['yes family history'] = df['family history'] == 'yes'
df['488G'] == 'G'
df['488GGENE'] = df['488G'] == 'G'
df['488A'] == 'A'
df['488AGENE'] = df['488A'] == 'A'
df['308A'] == 'A'
df['308AGENE'] = df['308A'] == 'A'
df['308G'] == 'G'
df['308GGENE'] = df['308G'] == 'G'
df['380G'] == 'G'
df['380GGENE'] = df['380G'] == 'G'
df['380A'] == 'A'
df['380AGENE'] = df['380A'] == 'A'
df['859C'] == 'C'
df['859CGENE'] = df['859C'] == 'C'
df['859T'] == 'T'
df['859TGENE'] = df['859T'] == 'T'
df['3W-A'] == 'A'
df['3W-AGENE'] = df['3W-A'] == 'A'
df['3M-G'] == 'G'
df['3M-GGENE'] = df['3M-G'] == 'G'
df['1032 C'] == 'C'
df['1032 CGENE'] = df['1032 C'] == 'C'
df['1032T'] == 'T'
df['1032TGENE'] = df['1032T'] == 'T'
x = df[['age','married','yes family history','488GGENE','488AGENE','308AGENE','308GGENE','380GGENE','380AGENE','859CGENE','859TGENE','3W-AGENE','3M-GGENE','1032 CGENE','1032TGENE']].values
y = df['cancer'].values
model = LogisticRegression()
model.fit(x,y)
y_pred = model.predict(x)
model.score(x, y)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print('accuracy:', accuracy_score(y, y_pred))
print('precision:', precision_score(y, y_pred))
print('recall:', recall_score(y, y_pred))
print('f1 score:', f1_score(y, y_pred))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y, y_pred))
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y)

print('whole dataset:', x.shape, y.shape)
print('training set:', x_train.shape, y_train.shape)
print('test set:', x_test.shape, y_test.shape)

model = LogisticRegression()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))




import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_recall_fscore_support

sensitivity_score = recall_score
def specificity_score(y_true, y_pred):p,r,f,s = precision_recall_fscore_support(y_true,y_pred)
 
df = pd.read_csv('C:\\Users\\Alamukii\\Desktop\\joined data.csv')
df['marital status'] == 'married'
df['married'] = df['marital status'] == 'married'
df['family history'] == 'yes'
df['yes family history'] = df['family history'] == 'yes'
x = df[['age','married','yes family history']].values
y = df['cancer'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 5)

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('sensitivity:', sensitivity_score(y_test, y_pred))
print ('specificity:',specificity_score(y_test, y_pred))
model.predict_proba(x_test)
y_pred = model.predict_proba(x_test)[:,1]>0.75
print(y_pred)
print('precision:', precision_score(y_test, y_pred))
print('recall:',recall_score(y_test,y_pred))


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred_proba = model.predict_proba(x_test)
fpr, tpr, thresholds = roc_curve(y_test,y_pred_proba[:,1])
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1], linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('FALSE POSITIVE RATE (1-specificity)')
plt.ylabel('TRUE POSITIVE RATE(sensitivity)')
plt.show()


# In[ ]:




