# this file is used for getting the result from evaluate exe so as to test more efficiently

import subprocess
import pandas as pd


def f1_score(predictedTestResult):

    df_testOutput = pd.DataFrame(range(342999, 381109), columns=["id"])
    df_testOutput.insert(1, "Response", predictedTestResult, True)
    df_testOutput.to_csv("evaluate.csv", index=False)
    # handle the case where error in exe caused by zero division
    try:
        output = subprocess.check_output('evaluate_1.exe evaluate.csv').decode()
    except subprocess.CalledProcessError:
        output = 0.0

    return float(output)
