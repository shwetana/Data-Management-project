from pandas_profiling import ProfileReport
import pandas as pd
import numpy as np
from spellchecker import SpellChecker

#input file
input_file="Credit_Card_Customers.csv"

#staging file
preprocessed_file="Preprocessed_Data.csv"

#output file
customer_fact="Customer_Fact.csv"
income_dimension="Income_Dimension.csv"
education_dimension="Education_Dimension.csv"
gender_dimension="Gender_Dimension.csv"
marital_dimension="Marital_Dimension.csv"
card_dimension="Card_Dimension.csv"

#Data Profiling
#Creation of ETL Job
#Extraction
df_input=pd.read_csv(input_file)
prof=ProfileReport(df_input)
prof.to_file(output_file="profile_file_bank_churners_creditcard.html")

#===================================================================================
# Cleaning data - Transformation

#===================================================================================
#Validity Dimension
#===================================================================================
# used absolute value to make negative value-> in valid age range
df_input["Customer_Age"]=abs(df_input["Customer_Age"])

#===================================================================================
# Uniqueness Dimension
#===================================================================================
df_input.drop_duplicates(subset="CLIENTNUM",keep="first",inplace=True)
#===================================================================================
# Accuracy Dimension
#===================================================================================

#Typo Mistake
spell=SpellChecker()
df_input["Gender"]=df_input["Gender"].replace("Femle",spell.correction("Femle").title())

#Imputation techniques perform
df_input["Card_Category"]=df_input["Card_Category"].replace('Unknown',df_input["Card_Category"].mode())
df_input["Education_Level"]=df_input["Education_Level"].replace(['Unknown','shdjafs','ghjefs'],df_input["Education_Level"].mode()[0])
df_input["Income_Category"]=df_input["Income_Category"].replace('Unknown',df_input["Income_Category"].mode()[0])
df_input["Marital_Status"]=df_input["Marital_Status"].replace('Unknown',df_input["Marital_Status"].mode()[0])

#====================================================================================
#Consistency Dimension
#===================================================================================

## Title casing of column values
df_input["Gender"]=df_input["Gender"].str.title()
df_input["Education_Level"]=df_input["Education_Level"].str.title()
df_input["Card_Category"]=df_input["Card_Category"].str.title()

# Removing leading spaces
df_input["Card_Category"]=df_input["Card_Category"].str.strip()

## Making consistent Gender type representation in Gender Column
df_input["Gender"]=df_input["Gender"].replace({"Female":"F","Male":"M"})


#===================================================================================
# Completeness Dimension
#===================================================================================
df_input['Customer_Age']=df_input.Customer_Age.fillna(df_input.Customer_Age.mean())
df_input.dropna(axis="rows",inplace=True)

print("Total missing value in column:")
print(df_input.isna().sum())


#===================================================================================
#Staging Area
#===================================================================================

df_input.to_csv(preprocessed_file,encoding='utf-8',index=False)

#===================================================================================
# # Transformation
#===================================================================================
df_preprossed=pd.read_csv(preprocessed_file,header=0)

# creating mappers and creating Id columns for  dimensions
Education_Level_map = {
    'Uneducated'    : 0,
    'High School'   : 1,
    'College'       : 2,
    'Graduate'      : 3,
    'Post-Graduate' : 4,
    'Doctorate'     : 5
}

df_Education_Level_ID= df_preprossed['Education_Level'].map(Education_Level_map)
df_preprossed['Education_Level_Id']=pd.to_numeric(df_Education_Level_ID,downcast='integer')


Income_Category_map = {
    'Less than $40K' : 0,
    '$40K - $60K'    : 1,
    '$60K - $80K'    : 2,
    '$80K - $120K'   : 3,
    '$120K +'        : 4
}
df_Income_Category_ID= df_preprossed['Income_Category'].map(Income_Category_map)
df_preprossed['Income_Category_Id']=pd.to_numeric(df_Income_Category_ID,downcast='integer')


print(df_preprossed['Card_Category'].unique())

Card_Category_map = {
    'Blue'     : 0,
    'Silver'   : 1,
    'Gold'     : 2,
    'Platinum' : 3
}
df_Card_Category_ID= df_preprossed['Card_Category'].map(Card_Category_map)
df_preprossed['Card_Category_Id']=pd.to_numeric(df_Card_Category_ID,downcast='integer')

Attrition_Flag_map = {
    'Existing Customer' : 0,
    'Attrited Customer' : 1
}
df_preprossed['Attrition_Flag']= df_preprossed['Attrition_Flag'].map(Attrition_Flag_map)

Gender_Flag_map = {
    'F' : 0,
    'M' : 1
}
df_Gender_ID= df_preprossed['Gender'].map(Gender_Flag_map)
df_preprossed['Gender_Id']=pd.to_numeric(df_Gender_ID,downcast='integer')

Marital_Flag_map={
     'Single':0,
    'Married' :1,
    'Divorced':2
}

df_Marital_Status_ID= df_preprossed['Marital_Status'].map(Marital_Flag_map)

df_preprossed['Marital_Status_Id']=pd.to_numeric(df_Marital_Status_ID,downcast='integer')
df_preprossed["Customer_Age"]=pd.to_numeric(df_preprossed["Customer_Age"],downcast='integer')


#fact and dimension dataframes
df_Customer=df_preprossed.loc[:,['CLIENTNUM','Income_Category_Id','Education_Level_Id','Card_Category_Id','Marital_Status_Id','Attrition_Flag','Customer_Age','Months_on_book','Avg_Utilization_Ratio','Total_Ct_Chng_Q4_Q1','Total_Trans_Ct','Total_Trans_Amt','Total_Amt_Chng_Q4_Q1','Avg_Open_To_Buy','Total_Revolving_Bal','Credit_Limit','Contacts_Count_12_mon','Months_Inactive_12_mon','Total_Relationship_Count']]
df_Income_Category=df_preprossed.loc[:,['Income_Category_Id','Income_Category']]
df_Education_Level=df_preprossed.loc[:,['Education_Level_Id','Education_Level']]
df_Card=df_preprossed.loc[:,['Card_Category_Id','Card_Category']]
df_Marital_Status=df_preprossed.loc[:,['Marital_Status_Id','Marital_Status']]
df_Gender=df_preprossed.loc[:,['Gender_Id','Gender']]

# sorting and dropping the duplicates
df_customer_fact = df_Customer.sort_values(by = ['CLIENTNUM'], ascending = True, na_position = 'last')
df_income_dimension=df_Income_Category.sort_values(by = ['Income_Category_Id'], ascending = True, na_position = 'last').drop_duplicates(['Income_Category_Id'],keep = 'first')
df_education_dimension=df_Education_Level.sort_values(by = ['Education_Level_Id'], ascending = True, na_position = 'last').drop_duplicates(['Education_Level_Id'],keep = 'first')
df_gender_dimension=df_Gender.sort_values(by='Gender_Id', ascending = True, na_position = 'last').drop_duplicates(['Gender_Id'],keep = 'first')
df_marital_dimension=df_Marital_Status.sort_values(by = ['Marital_Status_Id'], ascending = True, na_position = 'last').drop_duplicates(['Marital_Status_Id'],keep = 'first')
df_card_dimension=df_Card.sort_values(by = ['Card_Category_Id'], ascending = True, na_position = 'last').drop_duplicates(['Card_Category_Id'],keep = 'first')

# Export the Star Schema and save them in Excel File respectively
df_customer_fact.to_csv(customer_fact, index = False)
df_income_dimension.to_csv(income_dimension, index = False)
df_education_dimension.to_csv(education_dimension, index = False)
df_gender_dimension.to_csv(gender_dimension,index = False)
df_marital_dimension.to_csv(marital_dimension, index = False)
df_card_dimension.to_csv(card_dimension, index = False)

#===================================================================================
#Machine Learning

#problem Statement: A manager at the bank is facing a problem related to customer attrition.
# A bank wants to use the data to predict a customer who is going to get churned,
# so they convince their customer to provide better credit card service and change customer's decision.
# (GOAL is to build a model which can predict churning customers)

#===================================================================================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, mean_squared_error
from sklearn.metrics import classification_report, plot_confusion_matrix,confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
from imblearn.combine import SMOTETomek


# Loading/Reading schema file
trainset = pd.read_csv(customer_fact)

#####==========================================To see correlation of features===========================================
# plt.figure(figsize=(16,12))
# sns.heatmap(trainset.corr())
# plt.show()

###=====================================================================================================================
                               #feature selection using correlation
####=======================================================================================
## Concept behind this step - Assume correlation coefficient between two features is 0.85.
# This means that 85% of the time, you can predict feature 2 just using the values of feature 1.
# In other words,if you have feature 1 in your dataset, feature 2 won’t bring much new information.
# That’s why there is no point in keeping feature 2 since it only adds to
# complexity when training a model.

def check_correlated_columns(df , threshold) :
    # A function to identify highly correlated features which have correaltion more then 90% and create a drop column list.
    # Compute correlation matrix with absolute values
    # print(df.corr().abs())
    matrix = df.corr().abs()
    # Create a boolean mask
    mask = np.triu(np.ones_like(matrix , dtype=bool))
    # Subset the matrix
    reduced_matrix = matrix.mask(mask)
    # Find columns that meet the threshold
    to_drop = [c for c in reduced_matrix.columns if \
               any(reduced_matrix[c] > threshold)]
    return to_drop

to_drop = check_correlated_columns(trainset, threshold=.9)
print(f'columns to drop-{to_drop}')
datafeatures_reduced = trainset.drop(to_drop, axis=1)
print(datafeatures_reduced.columns)


y = datafeatures_reduced['Attrition_Flag']
datafeatures_reduced.drop(columns=['Attrition_Flag'], inplace=True)
datafeatures_reduced.drop(columns=['CLIENTNUM'], inplace=True)


## checking for balance data present or not

print(trainset['Attrition_Flag'].value_counts())
# 0    8463
# 1    1620

## balancing the dataset

smk=SMOTETomek()
X_res,Y_res=smk.fit_resample(datafeatures_reduced,y)
print(f" x-shape={X_res.shape} and y-shape={Y_res.shape}")


# # split training, testing dataset- train == 80%, test == 20%
X_train, X_test, y_train, y_test = train_test_split(X_res,
                                                   Y_res,
                                                   test_size=0.2,
                                                   random_state=13)

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++======")


def logistic_regression():
    from sklearn.linear_model import LogisticRegressionCV
    model = LogisticRegressionCV(max_iter=1000)
    # print(model.get_params())
    model.fit(X_train, y_train)
    return model

def decision_tree():
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(max_depth=20,random_state=154)
    # print(model.get_params())
    model.fit(X_train, y_train)
    return model

def random_forest():
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=13,n_estimators=100,max_depth=50)
    # print(model.get_params())
    model.fit(X_train, y_train)
    return model


def knn(metric):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(metric=metric, n_neighbors=5)
    print(model.get_params())
    model.fit(X_train, y_train)
    return model


def svm(kernel):
    from sklearn.svm import SVC
    model = SVC(kernel=kernel)
    # print(model.get_params())
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, model_name):
    y_pred = model.predict(X_test)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    model_acc = model.score(X_train , y_train)
    recall = recall_score(y_true=y_test , y_pred=y_pred)
    precision = precision_score(y_true=y_test , y_pred=y_pred)
    mse = mean_squared_error(y_true=y_test , y_pred=y_pred)

    print(f'Model: {model_name}')
    print('-' * 50)
    print(f'Model Accuracy: {model_acc:.4f}, Testing Accuracy: {accuracy:.4f}')
    print(f'Recall: {recall:.4f}, Precision: {precision:.4f}, MSE: {mse:.4f}')
    print(f"accuracy of {model_name} is {accuracy * 100:0.2f}%")
    print('-' * 50)

def visualize_result(model):
    from sklearn.tree import export_graphviz

    # creates a file named tree.dot
    export_graphviz(model, out_file="tree.dot", feature_names=X_train.columns, class_names='target', filled=True)


model_lr = logistic_regression()
evaluate_model(model_lr, "Logistic Regression")

model_knn_min = knn("minkowski")
evaluate_model(model_knn_min, "KNN")

model_knn_ed = knn("euclidean")
evaluate_model(model_knn_ed, "KNN")

model_knn_mh = knn("manhattan")
evaluate_model(model_knn_mh, "KNN")

model_dt = decision_tree()
evaluate_model(model_dt, 'Decision Tree')
visualize_result(model_dt)
# print('-' * 40)

model_dt = random_forest()
evaluate_model(model_dt, 'Random Forest')
# print('-' * 40)

###=============code hangs the system==================================================================================
# kernels = ['linear', 'poly']
# for kernel in kernels:
#     model_svm = svm(kernel)
#     evaluate_model(model_svm, f"SVM - {kernel}")


#####==================================================================================================================
## Model accuracy
####===================================================================================================================
# Model: Logistic Regression
# --------------------------------------------------
# Model Accuracy: 0.8412, Testing Accuracy: 0.8498
# Recall: 0.8468, Precision: 0.8590, MSE: 0.1502
# accuracy of Logistic Regression is 84.98%
# --------------------------------------------------
# {'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None,
# 'n_jobs': None, 'n_neighbors': 5, 'p': 2, 'weights': 'uniform'}
# Model: KNN
# --------------------------------------------------
# Model Accuracy: 0.9385, Testing Accuracy: 0.9237
# Recall: 0.9610, Precision: 0.8978, MSE: 0.0763
# accuracy of KNN is 92.37%
# --------------------------------------------------
# {'algorithm': 'auto', 'leaf_size': 30, 'metric': 'euclidean', 'metric_params': None,
# 'n_jobs': None, 'n_neighbors': 5, 'p': 2, 'weights': 'uniform'}
# Model: KNN
# --------------------------------------------------
# Model Accuracy: 0.9385, Testing Accuracy: 0.9237
# Recall: 0.9610, Precision: 0.8978, MSE: 0.0763
# accuracy of KNN is 92.37%
# --------------------------------------------------
# {'algorithm': 'auto', 'leaf_size': 30, 'metric': 'manhattan', 'metric_params': None,
# 'n_jobs': None, 'n_neighbors': 5, 'p': 2, 'weights': 'uniform'}
# Model: KNN
# --------------------------------------------------
# Model Accuracy: 0.9423, Testing Accuracy: 0.9267
# Recall: 0.9657, Precision: 0.8992, MSE: 0.0733
# accuracy of KNN is 92.67%
# --------------------------------------------------
# Model: Decision Tree
# --------------------------------------------------
# Model Accuracy: 0.9996, Testing Accuracy: 0.9535
# Recall: 0.9545, Precision: 0.9550, MSE: 0.0465
# accuracy of Decision Tree is 95.35%
# --------------------------------------------------
# Model: Random Forest
# --------------------------------------------------
# Model Accuracy: 1.0000, Testing Accuracy: 0.9811
# Recall: 0.9864, Precision: 0.9772, MSE: 0.0189
# accuracy of Random Forest is 98.11%
# --------------------------------------------------

## By seeing to 98 % accuracy it can be best model for predicting churning of credit card bank customers