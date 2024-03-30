import pandas as pd
import math

df = pd.read_csv("iris.csv")

# ------ #

# 1. Shuffle the dataframe
df_shuffled = df.sample(frac=1).reset_index(drop=True)

# 2. Split the dataframe into training and testing data ( 80/20 )
df_train = df_shuffled[0:120].copy()
df_test = df_shuffled[120:].copy()

df_test["Species"] = pd.NA # Clean last column

# 3. 

def distanciaEuclidiana (linha1, linha2):
    sl = (linha1['SepalLengthCm'] - linha2['SepalLengthCm']) ** 2
    sw = (linha1['SepalWidthCm'] - linha2['SepalWidthCm']) ** 2
    pl = (linha1['PetalLengthCm'] - linha2['PetalLengthCm']) ** 2
    pw = (linha1['PetalWidthCm'] - linha2['PetalWidthCm']) **2
    return math.sqrt(sl + sw + pl + pw)

print(df_test)