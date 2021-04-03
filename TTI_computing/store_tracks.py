import json
import pandas as pd
import pickle

if __name__ == "__main__":
    num_of_cars=200000

    df=pd.read_csv("E:/didi/城市交通指数和轨迹数据_2018/data/chengdushi_1001_1010.csv", nrows=num_of_cars, header=0, names=["track"], usecols=[2],iterator=True)
    # df=pd.read_csv("./data/chengdushi_1001_1010.csv",iterator=True)
    cnt = 0
    for times in range(20):
        sep_df = df.get_chunk(10000)
        track = []
        for temp in sep_df["track"]:
            temp = temp.lstrip("[").rstrip("]")
            temp = temp.split(", ")  # 注意分隔符是逗号+空格
            for i in range(len(temp)):
                temp[i] = temp[i].split(" ")
            for item in temp:
                item[0] = float(item[0])
                item[1] = float(item[1])
                item[2] = int(item[2])
            track.append(temp)
        with open("E:/didi/城市交通指数和轨迹数据_2018/data/track_" + str(cnt) + "_cars", "wb") as f:
            pickle.dump(track, f)
        cnt += sep_df.shape[0]
        print(times)

    exit(0)
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

    with open("./TrafficDataAnalysis/track_"+str(num_of_cars)+"_cars", "w") as f:
        pickle.dump(track, f)
    # with open("./TrafficDataAnalysis/track_"+str(num_of_cars)+"_cars.json", "w") as f:
    #     json.dump(track, f)

    print("Finished")