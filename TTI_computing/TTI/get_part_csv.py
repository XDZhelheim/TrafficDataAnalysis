import pandas as pd

# road = "D:\\innovative-practice2\\data\\boundary.txt"
# city = "D:\\innovative-practice2\\data\\city_district.txt"
path = "D:\\innovative-practice2\\data\\chengdushi_1001_1010.csv"


open_path = pd.read_csv(path,chunksize=100)

for i in open_path:
    print(i)
    t = i.to_csv("D:\\innovative-practice2\\data\\top100.csv")
    break
