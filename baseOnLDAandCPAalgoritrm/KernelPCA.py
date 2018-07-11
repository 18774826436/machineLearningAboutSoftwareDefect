# Importing the libraries
#数据预处理函数fit_transform()和transform()的区别
#fit_transform()的作用就是先拟合数据，然后转化它将其转化为标准形式
#即tranform()的作用是通过找中心和缩放等实现标准化


import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('cm1.csv')
X = dataset.iloc[:,:-1].values
X=X[400:498,:]
y = dataset.iloc[:, 21].values
y=y[400:498]
#然后需要创建包含自变量的矩阵和应变量的向量
#iloc表示取数据集中的某些行和某些列，逗号前表示行，逗号后表示列



# Encoding the Dependent Variable

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
#改变数据的形式，就我理解和数据的标准化差不多
#在运用一些机器学习算法的时候不可避免地要对数据进行特征缩放（feature scaling）
#，比如：在随机梯度下降（stochastic gradient descent）算法中，
#特征缩放有时能提高算法的收敛速度。
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying Dimensionality reduction
#本程序的核心，指定维度为6维，采用高斯核函数的进行区分，从而实现降为
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 6, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)



#############################################################
#对数据进行处理
#############################################################



# Fitting Decision Tree Classifier to the Training set
# 是一种无监督的学习方法，用于分类和回归。它对数据中蕴含的决策规则建模，
# 以预测目标变量的值。
#可以运用于二分类和多分类
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)





# Finding Area Under ROC curve
#AUC用于衡量“二分类问题”机器学习算法性能（泛化能力）。
#当测试集中的正负样本的分布变化的时候，ROC曲线能够保持不变。在实际的数据集中经常会出现类不平衡（class imbalance）现象，即负样本比正样本多很多（或者相反），
#而且测试数据中的正负样本的分布也可能随着时间变
#适合我们使用的数据集
from sklearn.metrics import roc_auc_score as roc
from sklearn.preprocessing import label_binarize
#print(y_test)
y_test= label_binarize(y_test,classes=[0,1])
y_pred= label_binarize(y_pred,classes=[0,1])

a = roc(y_test, y_pred, average='micro')

####################################################################

#Processing data for plotting graph
#将数据预处理为，相应合适的数据
from sklearn.preprocessing import label_binarize
y = label_binarize(y, classes=[0, 1])
n_classes = y.shape[1]

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
y = onehotencoder.fit_transform(y).toarray()

y_test = label_binarize(y_test, classes=[0, 1])
onehotencoder = OneHotEncoder(categorical_features = [0])
y_test = onehotencoder.fit_transform(y_test).toarray()


y_pred = label_binarize(y_pred, classes=[0, 1])
onehotencoder = OneHotEncoder(categorical_features = [0])
y_pred = onehotencoder.fit_transform(y_pred).toarray()

# Compute ROC curve and ROC area for each class
#画出ROC曲线，图像
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot of a ROC curve for a specific class
plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='red',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic jm1_kPCA')
plt.legend(loc="lower right")
plt.show()