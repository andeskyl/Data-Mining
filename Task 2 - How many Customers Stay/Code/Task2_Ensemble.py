import pandas as pd
from sklearn import preprocessing
from sklearn import ensemble
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
columnsIncluded = ["RowNumber","CreditScore","Geography","Age","Tenure","Balance",
                "NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]  # removed Surname, CustomerId

df_dataArray = df[columnsIncluded].copy()  # df without "Response"
dataArray = df_dataArray.values
dataArray = preprocessing.scale(dataArray)
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
dataArray_test = df_test[columnsIncluded].values
dataArray_test = preprocessing.scale(dataArray_test)

##########################################################
# train the model
numBaseClassifiers = 45
maxdepth = 4

# Random forest
clf = ensemble.RandomForestClassifier(n_estimators=numBaseClassifiers, random_state=1)
clf.fit(dataset['data'], dataset['target'])
predictedTestResult = clf.predict(dataArray_test)

# output to corresponding file
df_testOutput = df_test[["RowNumber"]]
df_testOutput.insert(1, "Exited", predictedTestResult, True)
df_testOutput.to_csv("C:/Users/Andes/Desktop/FTEC4003_Project/Task_2/Code/submission_2_Ensemble.csv", index=False)

# compute f1 score
f1_score_result = evaluateTask2.f1_score(predictedTestResult)
print("f1-score: " + str(f1_score_result))