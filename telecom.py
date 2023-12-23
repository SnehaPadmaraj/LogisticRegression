import os
os.getcwd()
os.chdir(r"C:\Users\sneha\Desktop\machine learning lab")
os.getcwd()
import pandas as pd
data = pd.read_csv("telecom.csv")
data.head()
data.isnull().sum()
import pingouin as pp 
pp.anova(data, dv= "Account length", between = "Churn")
pp.anova(data, dv= "Number vmail messages", between = "Churn")
pp.anova(data, dv= "Total day minutes", between = "Churn")
pp.anova(data, dv= "Total day calls", between = "Churn")
pp.anova(data, dv= "Total day charge", between = "Churn")
pp.anova(data, dv= "Total eve calls", between = "Churn")
pp.anova(data, dv= "Total eve minutes", between = "Churn")
pp.anova(data, dv= "Total eve charge", between = "Churn")
pp.anova(data, dv= "Total night minutes", between = "Churn")
pp.anova(data, dv= "Total night calls", between = "Churn")
pp.anova(data, dv= "Total intl calls", between = "Churn")
pp.anova(data, dv= "Total intl minutes", between = "Churn")
pp.anova(data, dv= "Total intl charge", between = "Churn")
pp.anova(data, dv= "Customer", between = "Churn")
data1 = data.drop(["Account length", "Total day calls", "Total eve calls", "Total night minutes", "Total night calls","Total intl calls", "Total intl minutes", "Total intl charge"], axis = 1)
data.head()
import scipy.stats
chisqt = pd.crosstab(data["Churn"], data["Voice mail plan"], margins = True)
chi2_stat, p , dof, expected = scipy.stats.chi2_contingency(chisqt)
p
chisqt1 = pd.crosstab(data["Churn"], data["International plan"], margins = True)
chi2_stat, p , dof, expected = scipy.stats.chi2_contingency(chisqt1)
p

data1["Churn"] = data1["Churn"].astype(str)
ratings = {"False":0, "True": 1}
data1["Churn"] = data1["Churn"].map(ratings)
y=data1["Churn"]
print(y.shape)
#y["Churn"] = y["Churn"].map(ratings)
#y.info()
#y = y.map(ratings)
x = data1.drop(["Churn"], axis = 1)
x = pd.get_dummies(x)
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
xtrain,xtest,ytrain,ytest = train_test_split(x,y, test_size = 0.20, random_state = 42)
model = LogisticRegression()
model.fit(xtrain,ytrain)
predicted_value = model.predict(xtest)
print(confusion_matrix(ytest, predicted_value))
print(accuracy_score(ytest, predicted_value))
