import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
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

df_dataArray = df[columnsIncluded].copy()  # df without "Response"
dataArray = df_dataArray.values
dataArray = minmax_scale(dataArray)  # map data to [0, 1]
target = df["Response"].tolist()

target_names = ["0", "1"]

dataset = {
"data": dataArray,
"target": target,
"feature_names": columnsIncluded,
"target_names": target_names
}

# load the test data
df_test = pd.DataFrame(data_test)
df_test.loc[df_test["Gender"]=="Male", "Gender"] = 0
df_test.loc[df_test["Gender"]=="Female", "Gender"] = 1
df_test.loc[df_test["Vehicle_Age"]=="< 1 Year", "Vehicle_Age"] = 0
df_test.loc[df_test["Vehicle_Age"]=="1-2 Year", "Vehicle_Age"] = 1
df_test.loc[df_test["Vehicle_Age"]=="> 2 Years", "Vehicle_Age"] = 2
df_test.loc[df_test["Vehicle_Damage"]=="No", "Vehicle_Damage"] = 0
df_test.loc[df_test["Vehicle_Damage"]=="Yes", "Vehicle_Damage"] = 1
dataArray_test = df_test[columnsIncluded].values
dataArray_test = minmax_scale(dataArray_test)  # map data to [0, 1]

##########################################################
# train the model
numBaseClassifiers = 1
maxdepth = 45

clf = ensemble.BaggingClassifier(DecisionTreeClassifier(max_depth=maxdepth), n_estimators=numBaseClassifiers)
clf.fit(dataset['data'], dataset['target'])
predictedTestResult = clf.predict(dataArray_test)

# output to corresponding file
df_testOutput = df_test[["id"]]
df_testOutput.insert(1, "Response", predictedTestResult, True)
df_testOutput.to_csv("submission_1_Ensemble.csv", index=False)

# compute f1 score
f1_score_result = evaluateTask1.f1_score(predictedTestResult)
print("f1-score: " + str(f1_score_result))
