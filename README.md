# Data Mining Project
The course project for FTEC4003 Data Mining, which is a competition on solving 2 classification problems.
## Document:
1. README.md
	- A file to describe group information and all the files
2. FTEC4003_report_06.pdf
	- A file to briefly describe the platform, the method, experimental evaluations, results and conclusions of the two tasks. 

## Package Imported:
- To install package, type the following command in a terminal:
```bash
pip install <package_name>
```
1. sklearn: responsible for implementing different data mining methods, as well as performing data preprocessing.
2. pandas: responsible for importing and exporting data from / to csv, as well as processing data
3. matplotlib: responsible for graph plotting for parameter optimization.
4. xgboost: responsible for the XGBoost Classifier in Task 2

## Code:
1. Task1_\<method\>.py
	- Python code of the method used in Task 1

2. evaluateTask1.py
	- Python code contains a function to get the result from evaluate_1.exe
	- Usage: Save evaluateTask1.py into the folder of task 1 and make sure the file contains evaluate_1.exe.
	- Type the following code to obtain f1-score after import evaluateTask1
```bash
f1_score_result = evaluateTask1.f1_score(predictedTestResult)
```

3. Task2_\<method\>.py
	- Python code of the method used in Task 2

4. evaluateTask2.py
	- Python code contains a function to get the result from evaluate_2.exe
	- Usage: Save evaluateTask2.py into the folder of task 2 and make sure the file contains evaluate_2.exe.
	- Type the following code to obtain f1-score after import evaluateTask2
```bash
f1_score_result = evaluateTask2.f1_score(predictedTestResult)
```

## Output:
1. submission_1_\<method\>.csv
	- Classification result from the method used in Task 1
2. submission_2_\<method\>.csv
	- Classification result from the method used in Task 2

## Achievement:
Best Project Award Runner-up in Fall 2020 FTEC4003.

© 2020 Team 2359
