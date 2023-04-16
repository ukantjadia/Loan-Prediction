# %% [markdown]
# ***CONTENTS*** <br>
# 
# - Intorduction
# 
# - Data Collection
# 
# - Data Preprocessing
# 
# - Data Splitting
# 
# - Model Implementation
# 
# - Model Evaluation

# %% [markdown]
# # Intorduction
# 
# In this project we have create a Machine Learning Model based on given information to predict whether or not loan will get approved.
# 
# 
# **What we will do?**
# 
# - Visualize and compare the data.
# 
# - Pre-processing of data.
# 
# - Handling Missing Value.
# 
# - Analyze Categorical and Numerical Data.
# 
# - Outliers Detection
# 
# - Different Machine Learning Algorithms and Evaluation Matrices for evaluation.
# 
# 
# **What we will Use?**
# 
# - Different Python Libraries such as `sklearn`,`matplotlib`,`numpy`,`seaborn`.
# 
# - Different Machine Learning Algorithm for Prediction Model and select best of them--
# 
#     - Logisctic Regression
#     
#     - KNeighbors Classifier
#     
#     - Support Vecort Machine(SVC)
#     
#     - DecisionTreeClassifier
#     
#  **NOTE:** I, Ukant doing this project under my Data Science Internship at CodeClause. Currently studying and learning about this field, so if there is any mistake I have made, please feel free comment below.

# %%
# Importing some libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings 
warnings.filterwarnings('ignore')

# %% [markdown]
# # Data Disscussion and Collection

# %%
df_train = pd.read_csv("../Data/train_u6lujuX_CVtuZ9i.csv")
df_test = pd.read_csv("../Data/test_Y3wMUE5_7gLdaTN.csv")

# %%
# Shape of data
print(df_train.shape)
print(df_test.shape)

# %%
df_train.head()

# %%
df_test.head()

# %%
# Stastical Summary of Continous data
df_train.describe()

# %% [markdown]
# - In above summary feature `Credit_History` only contain `0` and `1`,  so we need to change its type

# %%
# print(df_train.info())
print(df_test.info())

# %% [markdown]
# - we can see there is some values are missing in both object and float64 datatype.

# %% [markdown]
# # Data Preprocessing

# %%
# Statical summary of categorical Data
df_train['Credit_History'] = df_train['Credit_History'].astype('O')
df_train.describe(include='O')

# %%
# Checking for duplicate values
print(df_train.duplicated().sum())
print(df_test.duplicated().sum())

# %%
## Checking for null values
print(df_train.isnull().sum())
# print(df_test.isnull().sum())

# %%
## Let's analyze our traget feature 
plt.figure(figsize=(10,6))
sns.countplot(df_train['Loan_Status'])
plt.show()

print("The weight of Y class : %.2f" % (df_train['Loan_Status'].value_counts()[0] / len(df_train)*100))
print("The weight of N class : %.2f" % (df_train['Loan_Status'].value_counts()[1] / len(df_train)*100))

# %% [markdown]
# Bivariate Analysis

# %%
plt.figure(figsize=(10,6))
sns.countplot(x='Loan_Status',hue='Gender',data=df_train)
plt.title("Relationship between Loan Status and Gender",fontsize=18)
plt.show()

# %% [markdown]
# #### Observation
# - Most males got the more loans in comparision to females

# %%
plt.figure(figsize=(10,6))
sns.countplot(x='Credit_History',hue='Loan_Status',data=df_train)
plt.title("Relationship between Credit History and Loan Status",fontsize=18)
plt.show()

# %% [markdown]
# #### Observation
# - The more clear Credit History(1) more chance to get loan 
# - Not approving loan with credit history(0)

# %%
plt.figure(figsize=(10,6))
sns.countplot(x='Married',hue='Loan_Status',data=df_train)
plt.title("Relationship between Married status and Loan Status",fontsize=18)
plt.show()

# %% [markdown]
# #### Observation
# - Married people have better chance to get loan

# %%
grid = sns.FacetGrid(col='Loan_Status',data=df_train,size=3.5,aspect=1.5)
grid.map(sns.countplot,'Dependents')
plt.show()

# %% [markdown]
# #### Observation
# - Dependents with 1 have more chances to get loan

# %%
plt.figure(figsize=(10,6))
sns.countplot(x='Education',hue='Loan_Status',data=df_train)
plt.title("Relationship between Education and Loan Status",fontsize=18)
plt.show()


# %% [markdown]
# #### Observation
# - From above plot Graduate's have better chance of getting a loan

# %%
grid = sns.FacetGrid(col='Loan_Status',data=df_train,size=3.5,aspect=1.5)
grid.map(sns.countplot,'Self_Employed')
plt.show()

# %% [markdown]
# #### Observation
# - We can say, Self Employed people got more loan than others

# %%
grid = sns.FacetGrid(col='Loan_Status',data=df_train,size=3.5,aspect=1.5)
grid.map(sns.countplot,'Property_Area')
plt.show()

# %% [markdown]
# #### Observation
# - Here Semiurban Property Area get more loans in comparision to other area

# %%
plt.scatter(df_train['ApplicantIncome'],df_train['Loan_Status'])
plt.show()

# No Pattern 

# %% [markdown]
# Univariate Analysis

# %%
df_train.isnull().sum().sort_values(ascending=False)

# %%
## Dropping Loan Id 
df_train.drop('Loan_ID',axis=1,inplace=True)

# %%
## Separating the categorical and numerical data
cat_data = []
num_data = []

for name,dtype in enumerate(df_train.dtypes):
    if dtype == object:
        cat_data.append(df_train.iloc[:,name])
    else:
        num_data.append(df_train.iloc[:,name])

# %%
cat_data = pd.DataFrame(cat_data).T
num_data = pd.DataFrame(num_data).T

# %%
num_data

# %%
cat_data

# %%
## Handling missing values in categorical data
cat_data = cat_data.apply(lambda x: x.fillna(x.value_counts().index[0]))
cat_data.isnull().sum().sort_values(ascending=False)

# %%
## Handling missing values in numerical data
num_data.fillna(method='bfill',inplace=True)
num_data.isnull().sum().sort_values(ascending=False)

# %%
## Categorical Data Preprocessing

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# %%
target_values = {'Y':0,'N':1}
target = cat_data['Loan_Status']
cat_data.drop('Loan_Status',axis=1,inplace=True)
target = target.map(target_values)

# %%
for i in cat_data:
    cat_data[i] = le.fit_transform(cat_data[i])

# %%
cat_data

# %%
df = pd.concat([cat_data,num_data,target],axis=1)
df

# %% [markdown]
# # Data Splitting 

# %%
X = pd.concat([num_data,cat_data],axis=1)
y = target

# %%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# %%
print('X_test shape',X_test.shape)
print('X_train shape',X_train.shape)
print('y_test shape',y_test.shape)
print('y_train shape',y_train.shape)

# %% [markdown]
# # Model Implementation and Evaluation

# %%
## Various Machine Learning Algorithm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

models = {
    'LogisticRegression': LogisticRegression(random_state=42),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'SVC': SVC(random_state=42),
    'DecisionTreeClassifier':DecisionTreeClassifier(max_depth=1,random_state=42)
}

# %%
from sklearn.metrics import precision_score , recall_score, f1_score, log_loss, accuracy_score
def loss(y_true,y_pred,retu=False):
    pre = precision_score(y_true,y_pred)
    rec = recall_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred)
    loss = log_loss(y_true,y_pred)
    acc = accuracy_score(y_true,y_pred)

    if retu:
        return pre, rec, f1, loss, acc
    else:
        print(' pre: %.3f\n rec: %.3f\n f1: %.3f\n loss: %.3f\n acc: %.3f'% (pre,rec,f1,loss,acc))

# %%
def train_eval(models,X,y):
    for name, model in models.items():
        print(name, ":")
        model.fit(X,y)
        loss(y,model.predict(X))
        print('-'*10)

        
train_eval(models,X_train,y_train)

# %%
df_test

# %% [markdown]
# # For validation of program 

# %% [markdown]
# We have done the training and testing of our model with training data `df_train`. Now we have process the validation data and user input for prediction.

# %%
df_test.drop('Loan_ID',axis=1,inplace=True)
df_test

# %% [markdown]
# ***list of preprocessing we have used***
# - remove duplicate
# - seprate handle the missing value
# - transform the cat_data
# - --:transform target data
# - concat them 

# %%
# Handling duplicate values 
df_test.duplicated().sum()

# %%
## Changing the data type of `Credit History`
df_test['Credit_History'] = df_test['Credit_History'].astype('O')

# %%
## Seprating categorical and numerical data
Tcat_data = []
Tnum_data = []

for name, dtype in enumerate(df_test.dtypes):
    if dtype == object:
        Tcat_data.append(df_test.iloc[:,name])
    else:
        Tnum_data.append(df_test.iloc[:,name])

Tcat_data = pd.DataFrame(Tcat_data).T
Tnum_data = pd.DataFrame(Tnum_data).T

# %%
## Handling missing value in categorical data
Tcat_data = Tcat_data.apply(lambda x: x.fillna(x.value_counts().index[0]))
Tcat_data.isnull().sum()

# %%
## Handling missing value in numerical data
Tnum_data.fillna(method='bfill',inplace=True)
Tnum_data.isnull().sum()

# %%
Tcat_data

# %%
## Transforming the categorical data storing into other dataframe
Transform_cat_data = pd.DataFrame()
for data in Tcat_data:
    Transform_cat_data[data] = le.fit_transform(Tcat_data[data])

# %%
## Createing validation dataframe 
X_valid = pd.concat([Tnum_data,Transform_cat_data],axis=1)

# %%
## Predicting target with logisticRegression 
predict = models['LogisticRegression'].predict(X_valid)

# %%
output = pd.concat([Tnum_data,Tcat_data],axis=1)

# %%
## Collecting all validating data into one dataframe
predict = pd.DataFrame(predict)
output = pd.concat([output,predict],axis=1)
output = output.rename({0:'Predicted'},axis='columns')

# %%
output

# %%
## Saving validation file as output.csv
output.to_csv('../Data/output.csv')

# %% [markdown]
# ### Thank You :)
# - By Ukant Jadia [https://ukantjadia.me/linkedin](https://ukantjadia.me/linkedin)


