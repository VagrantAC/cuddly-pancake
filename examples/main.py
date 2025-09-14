from sklearn.datasets import load_iris

iris_dataset = load_iris()

print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))

# DESCR 数据的描述
print(iris_dataset['DESCR'][:193] + "\n...")

# target_names 花的种类
print("Target names: {}".format(iris_dataset['target_names']))

# 