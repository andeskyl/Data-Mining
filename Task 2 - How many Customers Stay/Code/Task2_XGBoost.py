import pandas as pd
from sklearn import preprocessing
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
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
columnsIncluded = ['CreditScore', 'Geography', 'Gender', 'Age', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']

df_dataArray = df[columnsIncluded].copy()  # df without "Response"
dataArray = preprocessing.scale(df_dataArray.values)
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

##########################################################
# Train the model
clf = XGBClassifier(n_estimators=217, max_depth=3, min_child_weight=1, gamma=0,
                    subsample=0.62, colsample_bytree=1, reg_alpha=0, reg_lambda=0.6, learning_rate=0.1)
clf = clf.fit(dataset['data'],dataset['target'])     # input the given dataset for training
predictedTestResult = clf.predict(preprocessing.scale(df_test[columnsIncluded].values))

# output to corresponding file
df_testOutput = df_test[["RowNumber"]]
df_testOutput.insert(1, "Exited", predictedTestResult, True)
df_testOutput.to_csv("submission_2_XGBoost.csv", index=False)

# compute f1 score
f1_score_result = evaluateTask2.f1_score(predictedTestResult)
print("f1-score: " + str(f1_score_result))
