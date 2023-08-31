import pandas as pd


data1 = pd.read_csv("data1.csv")
data2 = pd.read_csv("data2.csv")
data3 = pd.read_csv("data3.csv")
data4 = pd.read_csv("data4.csv")
data5 = pd.read_csv("data5.csv")
data6 = pd.read_csv("data6.csv")
data7 = pd.read_csv("data7.csv")

res = pd.concat([data1, data2, data3, data4, data5, data6, data7], axis=0)
res.to_csv("data.csv", index=False)
