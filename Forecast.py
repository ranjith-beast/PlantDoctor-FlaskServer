import pandas as pd
from sklearn import linear_model
dataset = pd.read_csv("apy.csv")
dataset = dataset.dropna()
crop=input("Enter crop name")
district=input("Enter the district name")

x=[]
y=[]
for row in dataset.iterrows():
  if row[1][1]==district and row[1][4]==crop:
    x.append(row[1][5])
    print(x)
    y.append(row[1][6])
    print(y)

x=dataset[dataset['Crop']==crop]['Area']
y=dataset[dataset['Crop']==crop]['Production']

train=pd.Series(x)
test=pd.Series(y)

regg=linear_model.LinearRegression()
regg.fit(train.values.reshape(-1,1),test.values.reshape(-1,1))

ar=float(input("Enter the area"))

coeff=regg.coef_
production=coeff*ar

print("Production will be approximately:",production)
