# FTEC4003 Course Project Task 1: Insurance Selling
## 1. Background
- This data comes from clients of an insurance company. These clients have already bought the medical insurance. Nowadays, the company wants to launch a new transportation insurance and to find those who will be interested in this insurance.
## 2. Data Set Information
- The data is related with an insurance selling problem. The clients' information is about clients' basic information and their vehicle's situations.
#### This data set contains two files:
1. insurance-train.csv
	- The training set with 11 input attributes and 1 output attribute (i.e. class attribute)
2. insurance-test.csv
	- The testing set with 11 input attributes. You need to identify the class of each item. 

#### Other files
1. samplesubmission.csv:
	- This is a sample file to show the output format. Wrong format will lead to unkown result.

2. evaluate_1.macOS:
	- This is a command line tool to evaluate your result. We will use F1-measure to measure your result.
	- Usage: Press "command + space" to open spotlight search and type in "terminal", then type in the following command in the terminal. You should replace
```./submission_1_method.csv``` with your own path to the submission_1_method.csv.
```bash
./evaluate_1.macOS ./submission_1_method.csv
```

3. evaluate_1.linux:
	- Usage: Press "ctrl + alt + t" to launch a terminal and input the following command.
```bash
./evaluate_1.linux ./submission_1_method.csv
```

4. evaluate_1.exe:
	- Usage: Press "command + r" and then type in "cmd" in the dialog box to launch a terminal. Then type in the command:
```bash
./evaluate_1.exe ./submission_1_method.csv
```

## 3. Goal

- The classification goal is to predict if the client will buy the transportation insurance (i.e, Identify the value of feature 'Response', 1 for yes and 0 otherwise).

## 4. Attribute Information
#### a) Input variables
**clients' basic information**
1. ID: Unique ID of clients (numeric)
2. Gender: Gender of clients (categorical: 'Male', 'Female')
3. Age: Age of clients (numeric)
4. Driving_License: whether the clients have a driving license (categorical: '0', '1')
5. Region_Code: Unique code for the region of the clients (numeric)
6. Previously_Insured: whether the clients have already a transportation insurance (categorical: '0', '1')

**clients' vehicle situations**
7. Vehicle_Age: Age of the Vehicle (string)
8. Vehicle_Damage: whether the vehicle has been damaged (categorical: 'No', 'Yes')

**other attributes**
9. Annual_Premium: The amount customer needs to pay as premium in the year (numeric)
10. Policy_Sales_Channel: Anonymised Code for the channel of outreaching to the customer ie. Different Agents, Over Mail, Over Phone, In Person, etc. Here, using unique numbers to represent different channel (numeric)
11. Vintage: Number of Days that Customer has been associated with the company (numeric)

#### b) Output variable
12. Response: whether the client is interested in it (categorical: '0', '1')