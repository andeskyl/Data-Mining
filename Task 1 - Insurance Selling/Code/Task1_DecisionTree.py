import pandas as pd
from sklearn import tree
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
columnsIncluded = ["id", "Gender", "Age", "Driving_License", "Region_Code", "Previously_Insured", "Vehicle_Age",
                 "Vehicle_Damage", "Annual_Premium", "Policy_Sales_Channel", "Vintage"]

df_dataArray = df[columnsIncluded].copy()  # df without "Response"
dataArray = df_dataArray.values
target = df["Response"].tolist()

target_names = ["0", "1"]

dataset = {
 "data": dataArray,
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
testArray = df_test[columnsIncluded].values

##########################################################
# Train the model
clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=121)
# criterion use "entropy" or "gini"; max_depth can use any integer > 0
clf = clf.fit(dataset['data'],dataset['target'])     # input the given dataset for training
predictedTestResult = clf.predict(testArray)

### output to corresponding file
df_testOutput = df_test[["id"]]
df_testOutput.insert(1, "Response", predictedTestResult, True)
df_testOutput.to_csv("submission_1_DecisionTree.csv", index=False)

# compute f1 score
f1_score_result = evaluateTask1.f1_score(predictedTestResult)
print("f1-score: " + str(f1_score_result))