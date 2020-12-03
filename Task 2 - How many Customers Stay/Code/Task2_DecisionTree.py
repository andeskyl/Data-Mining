import pandas as pd
from sklearn import tree
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
dataArray = (df_dataArray.values)
target = df["Exited"].tolist()

target_names = ["0", "1"]

dataset = {
"data": dataArray,
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
Array = (df_test[columnsIncluded].values)

##########################################################
# train the model
# we can use 'entropy' or 'gini'(default)
clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=7)  # can adjust max_depth #####
clf = clf.fit(dataset['data'],dataset['target'])     # input the given dataset for training

# predict and output the test result
predictedTestResult = clf.predict(Array)

df_testOutput = df_test[["RowNumber"]]
df_testOutput.insert(1, "Exited", predictedTestResult, True)
df_testOutput.to_csv("submission_2_DecisionTree.csv", index=False)

# compute f1 score
f1_score_result = evaluateTask2.f1_score(predictedTestResult)
print("f1-score: " + str(f1_score_result))