import pandas as pd
from sklearn import preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.neural_network import MLPClassifier
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
columnsIncluded = ["CreditScore","Geography","Gender","Age","Tenure","Balance",
                  "NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]  # removed RowNumber, Surname, CustomerId

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
# train the model
rbf_feature = RBFSampler(gamma=0.01, n_components=6000, random_state=1)
X_features = rbf_feature.fit_transform(dataset['data'])
clf = MLPClassifier(hidden_layer_sizes=(10,10,10),solver='adam', activation='tanh', max_iter=5000, random_state=1)
clf = clf.fit(X_features,dataset['target'])
predictedTestResult = clf.predict(rbf_feature.fit_transform(preprocessing.scale(df_test[columnsIncluded].values)))

# output to corresponding file
df_testOutput = df_test[["RowNumber"]]
df_testOutput.insert(1, "Exited", predictedTestResult, True)
df_testOutput.to_csv("submission_2_Kernel.csv", index=False)

# compute f1 score
f1_score_result = evaluateTask2.f1_score(predictedTestResult)
print("f1-score: " + str(f1_score_result))