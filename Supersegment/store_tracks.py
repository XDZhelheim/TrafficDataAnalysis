import json
import pandas as pd

if __name__ == "__main__":
    num_of_cars=20000

    df=pd.read_csv("./TrafficDataAnalysis/chengdushi_1001_1010.csv", nrows=num_of_cars, header=0, names=["track"], usecols=[2])
    track=[]
    for temp in df["track"]:
        temp=temp.lstrip("[").rstrip("]")
        temp=temp.split(", ") # 注意分隔符是逗号+空格
        for i in range(len(temp)):
            temp[i]=temp[i].split(" ")
        for item in temp:
            item[0]=float(item[0])
            item[1]=float(item[1])
            item[2]=int(item[2])
        track.append(temp)

    with open("./TrafficDataAnalysis/track_"+str(num_of_cars)+"_cars.json", "w") as f:
        json.dump(track, f)

    print("Finished")