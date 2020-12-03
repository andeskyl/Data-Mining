import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import evaluateTask1

# import csv data
data = pd.read_csv('insurance-train.csv')
data_test = pd.read_csv('insurance-test.csv')

df = pd.DataFrame(data)
# translate data
df.loc[df["Gender"]=="Male", "Gender"] = 0
df.loc[df["Gender"]=="Female", "Gender"] = 1
df.loc[df["Vehicle_Age"]=="< 1 Year", "Vehicle_Age"] = 0
df.loc[df["Vehicle_Age"]=="1-2 Year", "Vehicle_Age"] = 1
df.loc[df["Vehicle_Age"]=="> 2 Years", "Vehicle_Age"] = 2
df.loc[df["Vehicle_Damage"]=="No", "Vehicle_Damage"] = 0
df.loc[df["Vehicle_Damage"]=="Yes", "Vehicle_Damage"] = 1

# feature selection
columnsIncluded = ["Gender", "Age", "Driving_License", "Region_Code", "Previously_Insured", "Vehicle_Age",
                  "Vehicle_Damage", "Annual_Premium", "Policy_Sales_Channel", "Vintage"]

pca = PCA(n_components=8)
df_dataArray = df[columnsIncluded].copy()  # df without "Response"
dataArray = df_dataArray.values
#The first 3 lines are preprocessing the data using PCA,
# and the forth line is doing preprocessing.scale method to scale down the dataset.
x1 = dataArray.transpose()
x2 = pca.fit(x1)
x3 = x2.components_.transpose()
x4 = preprocessing.scale(x3)
target = df["Response"].tolist()

target_names = ["0", "1"]

dataset = {
  "data": x4,
  "target": target,
  "feature_names": columnsIncluded,
  "target_names": target_names
}

# Evaluate the model
# predict and output the test result
df_test = pd.DataFrame(data_test)
df_test.loc[df_test["Gender"]=="Male", "Gender"] = 0
df_test.loc[df_test["Gender"]=="Female", "Gender"] = 1
df_test.loc[df_test["Vehicle_Age"]=="< 1 Year", "Vehicle_Age"] = 0
df_test.loc[df_test["Vehicle_Age"]=="1-2 Year", "Vehicle_Age"] = 1
df_test.loc[df_test["Vehicle_Age"]=="> 2 Years", "Vehicle_Age"] = 2
df_test.loc[df_test["Vehicle_Damage"]=="No", "Vehicle_Damage"] = 0
df_test.loc[df_test["Vehicle_Damage"]=="Yes", "Vehicle_Damage"] = 1
dataArray_test = df_test[columnsIncluded].values
#The first 3 lines are preprocessing the data using PCA,
# and the forth line is doing preprocessing.scale method to scale down the dataset.
y1 = dataArray_test.transpose()
y2 = pca.fit(y1)
y3 = y2.components_.transpose()
y4 = preprocessing.scale(y3)

##########################################################
# Train the model
clf = KNeighborsClassifier(n_neighbors=2, metric='minkowski', weights='distance')
#n_neighbors can be any integer with n_neighbors>0, weights can be 'distance' or 'uniform'
clf.fit(dataset['data'],dataset['target'])
predictedTestResult = clf.predict(y4)

# output to corresponding file
df_testOutput = df_test[["id"]]
df_testOutput.insert(1, "Response", predictedTestResult, True)
df_testOutput.to_csv("submission_1_k-NearestNeighbor.csv", index=False)

# compute f1 score
f1_score_result = evaluateTask1.f1_score(predictedTestResult)
print("f1-score: " + str(f1_score_result))