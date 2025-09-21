from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import mglearn
import matplotlib.pyplot as plt

iris_dataset = load_iris()

print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))

# DESCR 数据的描述
print(iris_dataset["DESCR"][:193] + "\n...")

# target_names 花的种类
print("Target names: {}".format(iris_dataset["target_names"]))

# feature_name 特征的名字
print("Feature names: \n{}".format(iris_dataset["feature_names"]))

# 数据集的类型
print("Type of data: {}".format(type(iris_dataset["data"])))

# 数据集的大小(150条数据, 4个特征)
print("Shape of data: {}".format(iris_dataset["data"].shape))

# 前五行数据
print("First five rows of data:\n{}".format(iris_dataset["data"][:5]))

# target的数据类型
print("Type of target: {}".format(type(iris_dataset["target"])))

# target的数据大小(150条数据, 1个结果)
print("Shape of target: {}".format(iris_dataset["target"].shape))

# 用数字映射target结果
print("Target:\n{}".format(iris_dataset["target"]))

# 泛化能力：是指训练出的模型能否正确彻底预测

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset["data"], iris_dataset["target"], random_state=0
)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = pd.plotting.scatter_matrix(
    iris_dataframe,
    c=y_train,
    figsize=(15, 15),
    marker="o",
    hist_kwds={"bins": 20},
    s=60,
    alpha=0.8,
    cmap=mglearn.cm3,
)
# plt.show()



from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)
