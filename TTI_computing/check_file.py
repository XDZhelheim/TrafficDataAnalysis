import pandas as pd

path = "../../data/chengdushi_1001_1010.csv"

f = pd.read_csv(path, nrows=24)
print(f.head(4))