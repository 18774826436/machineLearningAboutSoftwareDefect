# machineLearningAboutSoftwareDefect
# Part 1(Use machineLearning)
## brief introduction

    A Method based on Oversampling and Ensemble Learning for Predicting the Number of Software Defects
### 运行环境介绍

    本项目是在anaconda3的win10电脑上开发出来的，若想看到本项目的具体效果，请安装相对应的环境，采用的编辑器是目前交互性极好的jupyter，
    若想打开正确浏览里面的代码和注释，强烈建议使用jupyter，其他的编辑器打开可能会出现问题。

### 本项目中使用的技术和库

    使用了python的一些常用库，包括主要用于科学计算的numpy和pandas,通过 IPython.display 和 matplotlib.pyplot，进行数据的可视化。
    根据我们样本的特点我们采用了sklearn中的BayesianRidge线性模型、DecisionTreeRegressor决策树回归模型和NearestNeighbors临近算法
    来训练我们的model。



# Part 2(use DeepLearning)

## ABSTRACT 

	Software Defect Prediction is an important aspect in order to ensure software quality. Deep Learning techniques can also be used for the same. In this paper, we propose to extract a set of expressive features from an initial set of basic change measures using Artificial Neural Network (ANN), and then train a classifier based on the extracted features using Decision tree and compare it to three other methods wherein features are extracted from a set of initial change measures using dimensionality reduction techniques that include Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA) and Kernel PCA. We use five open source datasets from NASA Promise Data Repository to perform this comparative study. For evaluation, three widely used metrics: Accuracy, F1 scores and Areas under Receiver Operating Characteristic curve are used. It is found that Artificial Neural Network outperformed all the other dimensionality reduction techniques. Kernel PCA performed best amongst the dimensionality reduction techniques.
	
  
## 项目简介

### 运行环境介绍

  第二部分，采用的是和第一部分完全不同的DeepLearning 中常常用于软件缺陷测试的LDA,CPA和KCPA算法，同样也是在win10，anaconda3下面中默认的SpyderIDE中开发的，主要是考虑到开发和调试的便利性，而且Spyder对于数据的可视化做的更加优秀，虽然体量较大，但是能够接受。

### 项目中使用的库和技术

  包括常用于科学计算的，numpy和pandas更多的是sklearn下的封装好的preprocessing（用于数据的预处理）；PCA（主成分分析）算法、LDC（Linear Discriminant Analysis）算法、KPCA（核主成分分析）算法，决策树分类算法（DecisionTreeClassifier），使用ROC curve曲线图来可视化算法的精确度。

1.使用pd.read_csv读入cm1.csv文件，该文件是NASA的中关于软件缺陷预测的中的datasets中的一个。

2.使用labelencoder对文件进行，预处理的编码过程。

3.接着使用train_test_split0函数来将原本是一个整体的文件按照25%的比例合理的分为测试集和训练集。
4.之后使用StandardScaler函数对数据进行进一步的规范化处理。

5.然后就是程序的核心，本部分一共提供了三种不同的算法来对，软件缺陷进行预测，比较出那种算法的优势和不足，并得出混淆矩阵。

6.利用sklearn.metrics中的roc_auc_score方法求出roc curve中的AUC

7.之后就是数据的可视化，先利用sklearn中的预处理方法如label_binarize、OneHotEncoder，对生成的数据进行处理。
	
8.同样使用python常用的matplotlib.pyplot可视化库，来可视化roc曲线，和对应的数据。
