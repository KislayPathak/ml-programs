import pandas as pd
import matplotlib.pyplot as plt

setd= pd.read_csv('D:/data/BC_data.csv')
x=setd.iloc[:,2:-1].values
y=setd.iloc[:,1].values

from sklearn.preprocessing import Imputer
imp= Imputer(missing_values='NaN',strategy ='mean')
imp=imp.fit(x[:,2:-1])
x[:,2:-1]=imp.transform(x[:,2:-1])


from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
y=label.fit_transform(y)



#trainn test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

from sklearn.linear_model import LogisticRegression
cla=LogisticRegression()
cla.fit(x_train,y_train)

y_pred=cla.predict(x_test)

from sklearn.metrics import confusion_matrix
res = confusion_matrix(y_test,y_pred)


plt.scatter(x_train[:,0],y_train,color='black')
plt.plot(x_train,cla.predict(x_train[:,0]),color='red')
plt.title('sal/exp')
plt.xlabel('exp')
plt.ylabel('sal')
plt.show()


plt.scatter(x_test,y_test,color='black')
plt.plot(x_test,cla.predict(x_test),color='red')
plt.title('sal/exp')
plt.xlabel('exp')
plt.ylabel('sal')
plt.show()
