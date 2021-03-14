import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gp
from shapely.geometry import Polygon, MultiLineString, Point
import shapely.wkt as wkt
import math
from sklearn.cluster import KMeans
import time
import datetime
from pyproj import CRS

colors=["#A1E2E6", "#E6BDA1", "#B3A16B", "#678072", "#524A4A"]

def get_tracks(num_of_cars: int) -> list:
    """
    获取 num_of_cars 辆车的轨迹

    [
        [
            [longitude, latitude, timestamp], 
            [longitude, latitude, timestamp],
            ...
        ],
        [
            [longitude, latitude, timestamp], 
            [longitude, latitude, timestamp],
            ...
        ],
        ...
    ]
    """
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
    return track

def show_district_and_road(df, district_line_number_start, district_line_number_end, road_line_number_start, road_line_number_end):
    """
    此方法用来画城市和道路的整体图

    boundary.txt 的 0~20 行是区域的边界信息, 21 及之后是道路

    Typical use case: show_district_and_road(df, None, 20, 21, None) 注意是闭区间
    """
    fig, ax=plt.subplots(figsize=(12, 8))
    df.loc[district_line_number_start:district_line_number_end].plot(ax=ax, color=colors[1], edgecolor="white", linewidth=0.3)
    plt.title("成都市", fontsize=30, fontname="Source Han Serif CN", color="black")
    ax.axis("off")
    fig.canvas.set_window_title(u'成都市')

    df.loc[road_line_number_start:road_line_number_end].plot(ax=ax, color=colors[4], linewidth=0.2)

    plt.show()

def show_geom(df, color, title):
    """
    画点

    df 为 geodataframe

    使用例:

    df_midpoint=gp.GeoDataFrame(geometry=midpointcoords)

    show_points(df_midpoint, "red", "中点")
    """
    fig, ax=plt.subplots(figsize=(12, 8))
    df.plot(ax=ax, color=color, markersize=0.2)
    plt.title(title, fontsize=15, fontname="Source Han Serif CN", color="black")
    ax.axis("off")
    fig.canvas.set_window_title(title)
    plt.show()

def kmeans(kmeans_input, k=None, plot=False):
    # TODO: 选择 k 的算法
    if not k:
        # 利用SSE选择k, 手肘法
        SSE=[]  # 存放每次结果的误差平方和
        for k in range(1, 9):
            estimator=KMeans(n_clusters=k)
            estimator.fit(kmeans_input)
            SSE.append(estimator.inertia_)

        diff=[]
        for i in range(len(SSE)-1):
            diff.append(SSE[i]-SSE[i+1])

        for i in range(len(diff)):
            if diff[i]<=diff[0]/20:
                k=i+1
                break

        if plot:
            X = range(1, 9)
            plt.subplot(221)
            plt.xlabel('k')
            plt.ylabel('SSE')
            plt.plot(X, SSE, 'o-')

    km_model=KMeans(n_clusters=k)
    km_model.fit(kmeans_input)

    # 分类信息
    y_pred = km_model.predict(kmeans_input)

    centers = km_model.cluster_centers_

    if plot:
        # 3d所有点
        ax=plt.subplot(222, projection = '3d')
        ax.scatter(kmeans_input[:, 0], kmeans_input[:, 1], kmeans_input[:, 2], c=y_pred, cmap=plt.cm.tab10)

        # 2d所有点
        plt.subplot(223)
        plt.scatter(kmeans_input[:, 0], kmeans_input[:, 1], c=y_pred, cmap=plt.cm.tab10, s=30)
        for i in range(k):
            plt.annotate(str(i+1), (centers[i, 0], centers[i, 1]))

        # 3d 中心点 s为大小
        ax=plt.subplot(224, projection = '3d')
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='*', s=80)

        # ax.axis("off")
        plt.show()

    return k, y_pred, centers

def read_boundary(timer=False):
    if timer:
        start_read=time.time()

    df=pd.read_table("./TrafficDataAnalysis/boundary.txt")
    df['geometry']=df['geometry'].apply(wkt.loads)
    df=gp.GeoDataFrame(df)
    df.crs=CRS("epsg:4326")

    if timer:
        end_read=time.time()
        print("读文件用时:", str(datetime.timedelta(seconds=end_read-start_read)))

    return df

def get_matched_points(tracks, roads, plot=False, timer=False):
    # FIXME: 因为 buffer() 的作用 一个点可能会匹配到多条路段上

    if plot:
        points_forplot=[] # 只用来画图

    if timer:
        start_gps=time.time()

    points=[[] for i in range(len(roads))]
    for track in tracks:
        for gps in track:
            p=Point(gps[0], gps[1])
            for i in range(len(roads)):
                road=roads.iloc[i]
                if road.contains(p):
                    points[i].append({"coord": p, "time": gps[2]})
                    if plot:
                        points_forplot.append(p)
    
    if timer:
        end_gps=time.time()
        print("GPS 点匹配用时:", str(datetime.timedelta(seconds=end_gps-start_gps)))
    
    # 画所有点
    if plot:
        df_track=gp.GeoDataFrame(geometry=points_forplot)
        show_geom(df_track, "black", "所有点")

    return points

def get_midpoints(points, plot=False, timer=False):
    # TODO: 中点二次匹配
    if plot:
        midpoints_forplot=[]

    if timer:
        start_mid=time.time()

    midpoints=[[] for i in range(len(points))]
    for i in range(len(points)):
        for j in range(len(points[i])-1):
            midpoint={}
            midpoint["coord"]=Point((points[i][j]["coord"].x+points[i][j+1]["coord"].x)/2, (points[i][j]["coord"].y+points[i][j+1]["coord"].y)/2)
            midpoint["speed"]=points[i][j]["coord"].distance(points[i][j+1]["coord"])/abs(points[i][j+1]["time"]-points[i][j]["time"])
            midpoints[i].append(midpoint)
            if plot:
                midpoints_forplot.append(midpoint)

    if timer:
        end_mid=time.time()
        print("计算中点用时:", str(datetime.timedelta(seconds=end_mid-start_mid)))

    if plot:
        # 画中点
        midpointcoords=[]
        for midpoint in midpoints_forplot:
            midpointcoords.append(midpoint["coord"])
        df_midpoint=gp.GeoDataFrame(geometry=midpointcoords)
        show_geom(df_midpoint, "red", "中点")

    return midpoints

def get_segment_centers(midpoints, k=None, kmeans_details_plot=False, timer=False):
    """
    获取 segment 的中心点

    [
        [
            [longitude, latitude, speed(relative)], 
            [longitude, latitude, speed(relative)],
            ...
        ],
        [
            [longitude, latitude, speed(relative)], 
            [longitude, latitude, speed(relative)],
            ...
        ],
        ...
    ]
    """
    if timer:
        start_kmeans=time.time()

    segment_centers=[]

    for i in range(len(midpoints)):
        kmeans_input=[]
        for midpoint in midpoints[i]:
            kmeans_input.append([midpoint["coord"].x, midpoint["coord"].y, midpoint["speed"]])
        kmeans_input=np.array(kmeans_input)

        k, y_pred, centers=kmeans(kmeans_input, k, kmeans_details_plot)

        segment_centers.append(centers)

        plt.scatter(kmeans_input[:, 0], kmeans_input[:, 1], c=y_pred, cmap=plt.cm.tab10, s=15)
        for j in range(k):
            plt.annotate("{}-{}".format(i+1, j+1), (centers[j, 0], centers[j, 1]))

    if timer:
        end_kmeans=time.time()
        print("KMeans 用时:", str(datetime.timedelta(seconds=end_kmeans-start_kmeans)))

    plt.show()

    return segment_centers

if __name__ == "__main__":
    start=time.time()

    df=read_boundary(timer=True)

    # 1. GPS 点匹配到道路上
    # XXX: 双向车道, 注意 buffer_distance 参数能否把两方向区分出来
    num_of_cars=2000 # 最后调整到了 20000

    buffer_distance=0.00004

    # 羊市街+西玉龙街
    roads=df.loc[(df["obj_id"]==283504) | (df["obj_id"]==283505) | (df["obj_id"]==283506), "geometry"].apply(lambda x: x.buffer(distance=buffer_distance))
    
    show_geom(gp.GeoDataFrame(geometry=roads), "blue", "road")

    # road_start_index=21
    # road_end_index=21

    # roads=df.loc[road_start_index:road_end_index, "geometry"].apply(lambda x: x.buffer(distance=buffer_distance))

    tracks=get_tracks(num_of_cars)

    points=get_matched_points(tracks, roads, plot=True, timer=True)

    # 2. 计算中点坐标和速度
    midpoints=get_midpoints(points, plot=True, timer=True)

    # 3. 将中点放入 kmeans 模型
    segment_centers=get_segment_centers(midpoints, timer=True)

    end=time.time()
    print("总用时:", str(datetime.timedelta(seconds=end-start)))

    print(np.array(segment_centers))

    """
    20000 辆车
    36940 个轨迹点
    4 个 segment
    经度划分 [104.06060, 104.06346] [104.06347, 104.06580] [104.06581, 104.06814] [104.06815 104.07049]
    [
        [1.04062012e+02 3.06669499e+01 2.92533993e-05]
        [1.04069519e+02 3.06670723e+01 1.63441321e-05]
        [1.04064867e+02 3.06670066e+01 2.96561530e-05]
        [1.04066795e+02 3.06670652e+01 3.61342084e-05]]

    新方法: 与老方法之间误差 0.0002 以内（应该是新方法更准）
    """
