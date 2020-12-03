import subprocess
import pandas as pd


def f1_score(predictedTestResult):

    df_testOutput = pd.DataFrame(range(7500,10000), columns=["RowNumber"])
    df_testOutput.insert(1, "Exited", predictedTestResult, True)
    df_testOutput.to_csv("evaluate.csv", index=False)
    # handle the case where error in exe caused by zero division
    try:
        output = subprocess.check_output('evaluate_2.exe evaluate.csv').decode()
    except subprocess.CalledProcessError:
        output = 0.0

    return float(output)
