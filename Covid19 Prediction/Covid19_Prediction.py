#Covid19 Prediction Analysis

#Importing Libraries
import pandas as pd
import matplotlib.pyplot as  plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,RobustScaler,PowerTransformer
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import pickle
import warnings
warnings.filterwarnings("ignore")

#1.Reading of Data
def load_data():
	pd.set_option('display.max_columns',None)
	df=pd.read_csv(r'D:\Velocity Corporate Training\Python\Databases\Healthcare Project\Covid19Precondition\covid.csv')
	print(df.head())
	print(df.info())
	print(df.describe())
	return df

#2. Data Preprocessing and Feature Engineering
def data_prec(df):
	#Feature Conversion into Proper Categories
	df['sex']=df['sex'].replace({1:'Female',2:'Male'})
	df['patient_type']=df['patient_type'].replace({1:'Outpatient',2:'Inpatient'})
	df['covid_res']=df['covid_res'].replace({1:'Positive',2:'Negative',3:'Results Awaited'})
	df['date_died']=df['date_died'].replace({'9999-99-99':'NA'})
	df.iloc[:,6:]=df.iloc[:,6:].replace([97,98,99],np.nan)
	df.iloc[:,6:]=df.iloc[:,6:].replace(1,'Yes')
	df.iloc[:,6:]=df.iloc[:,6:].replace(2,'No')

	#Dropping irrelavant Columns
	df = df.drop(df[df.covid_res=='Results Awaited'].index, axis=0)
	df=df.drop(['id','patient_type','entry_date','date_symptoms','date_died','intubed','tobacco','icu'],axis=1)
	print(df.head())

	#Chkecking of Null Values

	plt.figure(figsize=(10,8))
	sns.heatmap(df.isnull(),cmap='viridis')
	plt.title('Null value heatmap',size=25)
	print(df.isnull().sum())

	#Null Value Imputation
	df['pneumonia']=df['pneumonia'].fillna(df['pneumonia'].mode()[0])
	df['age']=df['age'].fillna(df['age'].mode()[0])
	df['pregnancy']=df['pregnancy'].fillna(df['pregnancy'].mode()[0])
	df['diabetes']=df['diabetes'].fillna(df['diabetes'].mode()[0])
	df['copd']=df['copd'].fillna(df['copd'].mode()[0])
	df['asthma']=df['asthma'].fillna(df['asthma'].mode()[0])
	df['inmsupr']=df['inmsupr'].fillna(df['inmsupr'].mode()[0])
	df['hypertension']=df['hypertension'].fillna(df['hypertension'].mode()[0])
	df['cardiovascular']=df['cardiovascular'].fillna(df['cardiovascular'].mode()[0])
	df['obesity']=df['obesity'].fillna(df['obesity'].mode()[0])
	df['renal_chronic']=df['renal_chronic'].fillna(df['renal_chronic'].mode()[0])
	df['contact_other_covid']=df['contact_other_covid'].fillna(df['contact_other_covid'].mode()[0])
	print(df.info())

	#Converting Categorical Columns into Numerical
	# Finding Categorical Columns
	categorical_feature_mask = df.dtypes==object

	# filter categorical columns using mask and turn it into a list
	categorical_cols = df.columns[categorical_feature_mask].tolist()
	print(categorical_cols)

	for c in categorical_cols:
		lbl = LabelEncoder() 
		lbl.fit(list(df[c].values)) 
		df[c] = lbl.transform(list(df[c].values))
		
	print(df.info())
		
	return df

#Classification Model Building

def class_model(df):
	#Data splitting train and test Data
	x = df.drop('covid_res', axis = 1)
	y = df.covid_res
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3)

	#Feature Scaling
	SC=StandardScaler()
	x_train=SC.fit_transform(x_train)
	x_test=SC.fit_transform(x_test)

	#Gradient Boosting classification Model without cross Validation
	model = GradientBoostingClassifier()
	model.fit(x_train, y_train)
	Grad_boost_pred = model.predict(x_test)
	Grad_boost_accuracy = metrics.accuracy_score(y_test, Grad_boost_pred)
	print("Accuracy: ",Grad_boost_accuracy)

	Grad_boost_precision=metrics.precision_score(y_test, Grad_boost_pred,pos_label=0)
	print("Precision: ",Grad_boost_precision)

	Grad_boost_recall=metrics.recall_score(y_test, Grad_boost_pred,pos_label=0)
	print("Recall: ",Grad_boost_recall)

	Grad_boost_f1_score= metrics.f1_score(y_test, Grad_boost_pred,pos_label=0)
	print("F1 Score: ",Grad_boost_f1_score)

	print("Confusion Matrix:\n",confusion_matrix(y_test,Grad_boost_pred))
	print("Classification Report:\n",classification_report(y_test,Grad_boost_pred))

	#Gradient Boosting classification Model with cross Validation
	#Grad_boost_cross_val = cross_val_score(model, x, y, cv=10, scoring='accuracy')
	#print('Classification Results with cross validation::')
	#Grad_boost_cv_accuracy = Grad_boost_cross_val.mean()
	#print("Accuracy: ",Grad_boost_cv_accuracy)

	#Grad_boost_cross_val_pre = cross_val_score(model, x, y, cv=10, scoring='precision_macro')
	#Grad_boost_cv_precision = Grad_boost_cross_val_pre.mean()
	#print("Precision: ",Grad_boost_cv_precision)

	#Grad_boost_cross_val_re = cross_val_score(model, x, y, cv=10, scoring='recall_macro')
	#Grad_boost_cv_recall = Grad_boost_cross_val_re.mean()
	#print("Recall: ",Grad_boost_cv_recall)

	#Grad_boost_cross_val_f1 = cross_val_score(model, x, y, cv=10, scoring='f1_macro')
	#Grad_boost_cv_f1_score = Grad_boost_cross_val_f1.mean()
	#print("F1 Score: ",Grad_boost_cv_f1_score)

	return model

#Driver Functions
#Calling of Data
df=load_data()

#Calling of Data Preprocessing, Feature Engineering and Model building
df=data_prec(df)
print(df.head(5))
GB=class_model(df)
print(GB)


#Pickling of file
file=open("Covid_Prediction.pkl","wb")
pickle.dump(GB,file)