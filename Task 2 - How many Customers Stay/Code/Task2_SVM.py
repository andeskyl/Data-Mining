import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
import evaluateTask2

# import csv data
data = pd.read_csv('train.csv')
data_test = pd.read_csv('assignment-test.csv')

df = pd.DataFrame(data)
# translate data
df.loc[df["Geography"]=="France", "Geography"] = 0
df.loc[df["Geography"]=="Spain", "Geography"] = 1
df.loc[df["Geography"]=="Germany", "Geography"] = 2
df.loc[df["Gender"]=="Male", "Gender"] = 0
df.loc[df["Gender"]=="Female", "Gender"] = 1

# feature selection
columnsIncluded = ["RowNumber","CustomerId","CreditScore","Geography","Gender","Age","Tenure","Balance",
                 "NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]  # removed Surname

df_dataArray = df[columnsIncluded].copy()  # df without "Response"
dataArray = df_dataArray.values
dataArray_scaled = preprocessing.scale(dataArray)
target = df["Exited"].tolist()

target_names = ["0", "1"]

dataset = {
"data": dataArray_scaled,
"target": target,
"feature_names": columnsIncluded,
"target_names": target_names
}

# predict and output the test result
df_test = pd.DataFrame(data_test)
df_test.loc[df_test["Geography"]=="France", "Geography"] = 0
df_test.loc[df_test["Geography"]=="Spain", "Geography"] = 1
df_test.loc[df_test["Geography"]=="Germany", "Geography"] = 2
df_test.loc[df_test["Gender"]=="Male", "Gender"] = 0
df_test.loc[df_test["Gender"]=="Female", "Gender"] = 1
df_test_Array = df_test[columnsIncluded].copy()
Array = df_test_Array[columnsIncluded].values
Array_scaled = preprocessing.scale(df_test_Array)

##########################################################
# train the model
clf = SVC(C=100,kernel='poly',gamma='auto', probability=False)
# C can be different number and C>0; kernel:{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’ and gamma:{‘scale’, ‘auto’} or float, default=’scale’
clf.fit(dataset['data'], dataset['target'])
predictedTestResult = clf.predict(Array_scaled)

# output to corresponding file
df_testOutput = df_test[["RowNumber"]]
df_testOutput.insert(1, "Exited", predictedTestResult, True)
df_testOutput.to_csv("submission_2_SVM.csv", index=False)

# compute f1 score
f1_score_result = evaluateTask2.f1_score(predictedTestResult)
print("f1-score: " + str(f1_score_result))