# FTEC4003 Course Project 
#### Document:
1. README.md
	- A file to describe group information and all the files
2. FTEC4003_report_06.pdf (not shown here)
	- A file to briefly describe the platform, the method, experimental evaluations, results and conclusions of the two tasks. 

#### Code:
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

#### Output:
1. submission_1_\<method\>.csv
	- Classification result from the method used in Task 1
2. submission_2_\<method\>.csv
	- Classification result from the method used in Task 2
