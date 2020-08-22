import os
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler,LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")

def train(df, x, y):
    global x_train, x_test, y_train, y_test
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)    

    #num_feat = [feature for feature in x.columns if x[feature].dtype != 'O']
    cat_feat = [feature for feature in x.columns if x[feature].dtype == 'O']

    col_transformer = ColumnTransformer(
                            transformers= [
                                            ('ordinal', OrdinalEncoder(), cat_feat)
											 
                                          ],
                                        remainder='drop',
                                        n_jobs=-1
                                        )

    #x_train_transformed = col_transformer.fit_transform(x_train)


    model = GradientBoostingClassifier()

    pipe = Pipeline([
                    ("preprocessing", col_transformer),
                    ("model", model)
                    ])
    pipe.fit(x_train, y_train)
    filename = 'Covid_Prediction.pkl'
    pickle.dump(pipe, open(filename, 'wb'))
    
    return filename


def test(filename):
    classifier = pickle.load(open(filename, 'rb'))
    test_data = x_test.head(1)
    pred = classifier.predict_proba(test_data)
    
    if pred >= 0.0 and pred<=0.25:
            print('Vulnerable Risk')
	elif pred > 0.25 and pred<=0.50:
		    print('Low Risk')
	elif pred > 0.50 and pred<=0.75:
		    print('Medium Risk')
    else:
            print('High Risk')
    
if __name__ == '__main__':
		
		df = pd.read_csv(r'C:\Users\DELL\Covid19 Prediction\assets\covid.csv')
	
		#converting numerical features to proper categories
		df['sex']=df['sex'].replace({1:'Female',2:'Male'})
		df['patient_type']=df['patient_type'].replace({1:'Outpatient',2:'Inpatient'})
		df['covid_res']=df['covid_res'].replace({1:'Positive',2:'Negative',3:'Results Awaited'})
		df['date_died']=df['date_died'].replace({'9999-99-99':'NA'})
		df.iloc[:,6:]=df.iloc[:,6:].replace([97,98,99],np.nan)
		df.iloc[:,6:]=df.iloc[:,6:].replace(1,'Yes')
		df.iloc[:,6:]=df.iloc[:,6:].replace(2,'No')
		

		df = df.drop(df[df.covid_res=='Results Awaited'].index, axis=0)
		df=df.drop(['id','patient_type','entry_date','date_symptoms','date_died','intubed','pregnancy','inmsupr','icu'],axis=1)
		
		#Null Value Handling
		df['pneumonia']=df['pneumonia'].fillna(df['pneumonia'].mode()[0])
		df['age']=df['age'].fillna(df['age'].mode()[0])
		df['diabetes']=df['diabetes'].fillna(df['diabetes'].mode()[0])
		df['copd']=df['copd'].fillna(df['copd'].mode()[0])
		df['asthma']=df['asthma'].fillna(df['asthma'].mode()[0])
		df['hypertension']=df['hypertension'].fillna(df['hypertension'].mode()[0])
		df['cardiovascular']=df['cardiovascular'].fillna(df['cardiovascular'].mode()[0])
		df['obesity']=df['obesity'].fillna(df['obesity'].mode()[0])
		df['renal_chronic']=df['renal_chronic'].fillna(df['renal_chronic'].mode()[0])
		df['tobacco']=df['tobacco'].fillna(df['tobacco'].mode()[0])
		df['contact_other_covid']=df['contact_other_covid'].fillna(df['contact_other_covid'].mode()[0])
		
		x = df.drop('covid_res',axis=1)
		y = df['covid_res']
		
		pickle_file_name = train(df, x, y)
		
		test(pickle_file_name)
