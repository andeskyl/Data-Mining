import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
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

df = df[:540]   # select a number of records to be used

# feature selection
columnsIncluded = ["Age", "Region_Code", "Previously_Insured", "Vehicle_Age",
                  "Vehicle_Damage", "Annual_Premium", "Vintage"]

df_dataArray = df[columnsIncluded].copy()  # df without "Response"
dataArray = df_dataArray.values
dataArray_scaled = preprocessing.scale(dataArray)

target = df["Response"].tolist()

target_names = ["0", "1"]

dataset = {
 "data": dataArray_scaled,
 "target": target,
 "feature_names": columnsIncluded,
 "target_names": target_names
}

# predict and output the test result
df_test = pd.DataFrame(data_test)
df_test.loc[df_test["Gender"]=="Male", "Gender"] = 0
df_test.loc[df_test["Gender"]=="Female", "Gender"] = 1
df_test.loc[df_test["Vehicle_Age"]=="< 1 Year", "Vehicle_Age"] = 0
df_test.loc[df_test["Vehicle_Age"]=="1-2 Year", "Vehicle_Age"] = 1
df_test.loc[df_test["Vehicle_Age"]=="> 2 Years", "Vehicle_Age"] = 2
df_test.loc[df_test["Vehicle_Damage"]=="No", "Vehicle_Damage"] = 0
df_test.loc[df_test["Vehicle_Damage"]=="Yes", "Vehicle_Damage"] = 1
df_test_Array = df_test[columnsIncluded].copy()
Array = df_test_Array[columnsIncluded].values
Array_scaled = preprocessing.scale(df_test_Array)

##########################################################
# Train the model
clf = SVC(C=10000, degree=3, kernel='poly', gamma='auto', probability=False, cache_size=1000)
clf.fit(dataset['data'], dataset['target'])
# C can be different number and C>0; kernel:{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’ and gamma:{‘scale’, ‘auto’} or float, default=’scale’
predictedTestResult = clf.predict(Array_scaled)

# output to corresponding file
df_testOutput = df_test[["id"]]
df_testOutput.insert(1, "Response", predictedTestResult, True)
df_testOutput.to_csv("submission_1_SVM.csv", index=False)

# compute f1 score
f1_score_result = evaluateTask1.f1_score(predictedTestResult)
print("f1-score: " + str(f1_score_result))