import numpy as np
import pandas as pd
pd.set_option('max_columns', None)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, SVC

def tr2(df):
  df = df.copy()
  for col in df.columns:
    if(df[col].dtype == np.number):
      continue
    df[col] = LabelEncoder().fit_transform(df[col])
  return df

def op_preprocess(sample,scaler):
    df = sample.copy()
    
    # Drop single-value columns and id columns
    df = df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1)

    # Binary-encode binary columns
    df['Gender'] = df['Gender'].replace({'Female': 0, 'Male': 1})
    df['OverTime'] = df['OverTime'].replace({'No': 0, 'Yes': 1})
      
    # Ordinal-encode the BusinessTravel column
    df['BusinessTravel'] = df['BusinessTravel'].replace({'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2})
    
    # One-hot encoding
    #for column in ['Department', 'EducationField', 'JobRole', 'MaritalStatus']:
        #df = onehot_encode(df, column=column)

    df = tr2(df)
    
    #print("Inside==============================================")
    #print(df.shape)
    
    # Split df into X and y
    #y = df['Attrition']
    #X = df.drop('Attrition', axis=1)
    
    # Train-test split
    #X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)
    
    # Scale X
    #scaler = StandardScaler()
    #scaler.fit(X_train)
    #X_test = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
    #X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
    
    return df


def pred2(svm):
  df = pd.read_csv(filename)
  df = op_preprocess(df,scaler)
  pred = svm.predict(df)
  #pred
  for row in range(df.shape[0]):
    for (intercept, coef) in zip(svm.intercept_, svm.coef_):
        s = "y = {0:.3f}".format(intercept)
        h = intercept
        mx = -1
        for (i, c) in enumerate(coef):
            s += " + {0:.3f} * x{1}".format(c, i)
            k = max(h,c*df.iloc[row][i])
            if(k>h):
              h = k
              mx = i
        #print(s)
        #print(h,mx)
        #print(pred[row],df.columns[mx])
        if(pred[row]==0):
          print("Don't Worry he will be with you")
        else:
          print("There is a high possibility that he will quit maybe due to",df.columns[mx])