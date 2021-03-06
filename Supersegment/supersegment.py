import datetime
import json
import os
import time

import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.wkt as wkt
from pyproj import CRS
from shapely.geometry import MultiPoint, Point, box
from sklearn.cluster import KMeans, MeanShift
from sklearn import metrics

# work directory
work_path="H:\\Codes\\PythonWorkspace\\TrafficDataAnalysis\\"

colors=["#A1E2E6", "#E6BDA1", "#B3A16B", "#678072", "#524A4A"]

def set_work_path(p: str):
    global work_path
    work_path=p

def show_district_and_road(df, district_line_number_start, district_line_number_end, road_line_number_start, road_line_number_end):
    """
    - 此方法用来画城市和道路的整体图
    - boundary.txt 的 0~20 行是区域的边界信息, 21 及之后是道路
    - Typical use case: show_district_and_road(df, None, 20, 21, None) 注意是闭区间
    """
    fig, ax=plt.subplots(figsize=(12, 8))
    df.loc[district_line_number_start:district_line_number_end].plot(ax=ax, color=colors[1], edgecolor="white", linewidth=0.3)
    plt.title("成都市", fontsize=30, fontname="Source Han Serif CN", color="black")
    ax.axis("off")
    fig.canvas.set_window_title(u'成都市')

    se=df.loc[road_line_number_start:road_line_number_end, "geometry"]

    # (104.0421, 30.6526) (104.1287, 30.7271)
    bound=gp.GeoSeries([box(104.0421, 30.6526, 104.1287, 30.7271)]) 
    se=se.append(bound)

    se.plot(ax=ax, color=colors[4], linewidth=0.2)

    plt.show()

def show_geom(obj, color, title, cmap=None, show=True, write_file=False):
    """
    - 画几何图形
    - obj 为 GeoDataFrame/GeoSeries
    - 使用例:
        - df_midpoint=gp.GeoDataFrame(geometry=midpointcoords)
        - show_points(df_midpoint, "red", "中点")
    """

    fig, ax=plt.subplots(figsize=(12, 8))
    obj.plot(ax=ax, color=color, cmap=cmap, markersize=0.2)
    plt.title(title, fontsize=15, fontname="Source Han Serif CN", color="black")
    ax.axis("off")
    fig.canvas.set_window_title(title)

    if show:
        plt.show()

    if write_file:
        plt.savefig(work_path+"supersegment_output/"+title+".png")

def kmeans(kmeans_input, k=None, metric=0, plot=False):
    # XXX: 选择 k 的算法

    best_k=1

    if not k:
        # 利用SSE选择k, 手肘法 
        if metric==0:
            SSE=[]  # 存放每次结果的误差平方和
            for k in range(1, 9):
                if len(kmeans_input)<k:
                    break

                estimator=KMeans(n_clusters=k)
                estimator.fit(kmeans_input)
                SSE.append(estimator.inertia_)

            diff=[]
            for i in range(len(SSE)-1):
                diff.append(SSE[i]-SSE[i+1])

            for i in range(len(diff)):
                best_k=i+2
                if diff[i]<=diff[0]/20:
                    break

            if plot:
                X=range(1, 9)
                plt.subplot(221)
                plt.xlabel('k')
                plt.ylabel('SSE')
                plt.plot(X, SSE, 'o-')

        # 使用 Calinski-Harabaz Index 评估, 越大越好 (不建议用)
        elif metric==1:
            best_k=0; best_score=-1
            for k in range(2, 9):
                if len(kmeans_input)<k:
                    break

                estimator=KMeans(n_clusters=k)
                estimator.fit(kmeans_input)
                score=metrics.calinski_harabasz_score(kmeans_input, estimator.labels_)

                if score>best_score:
                    best_k=k
                    best_score=score

            if plot:
                print("best_k = {}, best_score = {}".format(best_k, best_score))

    else:
        best_k=k

    km_model=KMeans(n_clusters=best_k)
    km_model.fit(kmeans_input)

    # 分类信息
    labels=km_model.labels_
    centers=km_model.cluster_centers_

    if plot:
        # 3d所有点
        ax=plt.subplot(222, projection = '3d')
        ax.scatter(kmeans_input[:, 0], kmeans_input[:, 1], kmeans_input[:, 2], c=labels, cmap=plt.cm.tab10)

        # 2d所有点
        plt.subplot(223)
        plt.scatter(kmeans_input[:, 0], kmeans_input[:, 1], c=labels, cmap=plt.cm.tab10, s=8)
        for i in range(best_k):
            plt.annotate(str(i+1), (centers[i, 0], centers[i, 1]))

        # 3d 中心点 s为大小
        ax=plt.subplot(224, projection = '3d')
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='*', s=80)

        # ax.axis("off")
        plt.show()

    return best_k, labels, centers

def read_boundary():
    df=pd.read_table(work_path+"boundary.txt")
    df['geometry']=df['geometry'].apply(wkt.loads)
    df=gp.GeoDataFrame(df)
    # df.crs=CRS("epsg:2432")

    return df

def get_tracks(num_of_cars=None, file_path=None, timer=True):
    """
    获取 num_of_cars 辆车的轨迹
    ```
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
    ```
    """

    if timer:
        start_read=time.time()

    if file_path:
        with open(file_path, "r") as f:
            tracks=json.load(f)
            tracks=list(tracks)

        return tracks

    df=pd.read_csv(work_path+"chengdushi_1001_1010.csv", nrows=num_of_cars, header=0, names=["track"], usecols=[2])
    tracks=[]
    for temp in df["track"]:
        temp=temp.lstrip("[").rstrip("]")
        temp=temp.split(", ") # 注意分隔符是逗号+空格
        for i in range(len(temp)):
            temp[i]=temp[i].split(" ")
        for item in temp:
            item[0]=float(item[0])
            item[1]=float(item[1])
            item[2]=int(item[2])
        tracks.append(temp)

    if timer:
        end_read=time.time()
        print("读轨迹用时:", str(datetime.timedelta(seconds=end_read-start_read)))

    return tracks

def get_matched_points(roads, tracks, plot=False, timer=True):
    """
    GPS 点匹配到路

    `points` 为一个四维数组
        - 第一维: 哪条路
        - 第二维: 哪条轨迹
        - 第三维: 哪个 GPS 点
        - 第四维: GPS 点内部
        - 实际上是个三维数组, GPS 应该是个 `tuple` 才对
        - 总之用的时候取到第三维，表示此轨迹在此条路上的一个 GPS 点
        - 增加轨迹那一维的原因是防止不同车的 GPS 点混在一起，不同车的轨迹点之间是不能求中点的 (时间段不同)

    设一个 GPS 点 `[longitude, latitude, timestamp]=[]`

    ```
    points = 
    [
        [
            [[], [], [], ...],
            [[], [], [], ...],
            ...
        ],
        [
            [[], [], [], ...],
            [[], [], [], ...],
            ...
        ],
        ...
    ]
    ```
    """
    # XXX: 因为 buffer() 的作用 一个点可能会匹配到多条路段上 -> 滤波(不要交叉), 按 distance 分配
    # FIXED: 最简单的方式 -> break
    # XXX: 时间复杂度优化, 存中间结果
    # FIXED: 两辆车不同时间经过同一段路

    if plot:
        points_forplot=[] # 只用来画图

    if timer:
        start_gps=time.time()

    minx, miny, maxx, maxy=roads.total_bounds

    points=[[[] for i in range(len(tracks))] for j in range(len(roads))]
    for i in range(len(tracks)):
        for gps in tracks[i]:

            # 框点，效果拔群
            if gps[0]<minx or gps[0]>maxx or gps[1]<miny or gps[1]>maxy:
                continue

            p=Point(gps[0], gps[1])
            for j in range(len(roads)):
                road=roads.iloc[j]
                if road.contains(p):
                    points[j][i].append({"coord": p, "time": gps[2]})
                    if plot:
                        points_forplot.append(p)
                    break
    
    if timer:
        end_gps=time.time()
        print("GPS 点匹配用时:", str(datetime.timedelta(seconds=end_gps-start_gps)))
    
    # 画所有点
    if plot:
        df_track=gp.GeoDataFrame(geometry=points_forplot)
        show_geom(df_track, "black", "所有点")

    return points

def get_midpoints(points, match=True, roads=None, plot=False, timer=True):
    """
    - match 表示是否只保留在路上的中点
    - 如果一条路是弧形，算出的中点可能不在路上
    """
    if plot:
        midpoints_forplot=[]

    if timer:
        start_mid=time.time()

    midpoints=[[] for i in range(len(points))]
    for i in range(len(points)):
        for track in points[i]:
            for j in range(len(track)-1):
                midpoint={}
                midpoint["coord"]=Point((track[j]["coord"].x+track[j+1]["coord"].x)/2, (track[j]["coord"].y+track[j+1]["coord"].y)/2)

                if match:
                    if not roads.iloc[i].contains(midpoint["coord"]):
                        continue
                
                midpoint["speed"]=track[j]["coord"].distance(track[j+1]["coord"])/abs(track[j+1]["time"]-track[j]["time"])
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

def cluster_kmeans(midpoints, k=None, kmeans_details_plot=False, timer=True, plot=True):
    """
    KMeans 算法获取 segment 的中心点
    ```
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
    ```
    """
    if timer:
        start_kmeans=time.time()

    midpoint_labels=[]
    segment_centers=[]

    for i in range(len(midpoints)):
        if not midpoints[i]: # 没有车经过这条路
            midpoint_labels.append(np.array([]))
            segment_centers.append(np.array([]))
            continue

        kmeans_input=[]
        for midpoint in midpoints[i]:
            kmeans_input.append([midpoint["coord"].x, midpoint["coord"].y, midpoint["speed"]])
        kmeans_input=np.array(kmeans_input)

        best_k, labels, centers=kmeans(kmeans_input, k, plot=kmeans_details_plot, metric=0)

        midpoint_labels.append(labels)
        segment_centers.append(centers)

        if plot and not kmeans_details_plot:
            # 调整不同配色
            if i%2:
                cmap=plt.cm.tab10
            else:
                cmap=plt.cm.Paired

            plt.scatter(kmeans_input[:, 0], kmeans_input[:, 1], c=labels, cmap=cmap, s=1)
            for j in range(best_k):
                plt.annotate("{}-{}".format(i+1, j+1), (centers[j, 0], centers[j, 1]))

    if timer:
        end_kmeans=time.time()
        print("KMeans 用时:", str(datetime.timedelta(seconds=end_kmeans-start_kmeans)))

    if plot and not kmeans_details_plot:
        plt.show()

    return midpoint_labels, segment_centers

def cluster_meanshift(midpoints, timer=True, plot=True):
    """
    MeanShift 算法获取 segment 的中心点
    ```
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
    ```
    """

    if timer:
        start_meanshift=time.time()

    midpoint_labels=[]
    segment_centers=[]

    ms_model=MeanShift(n_jobs=-1)

    for i in range(len(midpoints)):
        if not midpoints[i]: # 没有车经过这条路
            midpoint_labels.append(np.array([]))
            segment_centers.append(np.array([]))
            continue

        meanshift_input=[]
        for midpoint in midpoints[i]:
            meanshift_input.append([midpoint["coord"].x, midpoint["coord"].y, midpoint["speed"]])
        meanshift_input=np.array(meanshift_input)

        ms_model.fit(meanshift_input)

        labels=ms_model.labels_
        centers=ms_model.cluster_centers_

        midpoint_labels.append(labels)
        segment_centers.append(centers)

        if plot:
            # 调整不同配色
            if i%2:
                cmap=plt.cm.tab10
            else:
                cmap=plt.cm.Paired

            plt.scatter(meanshift_input[:, 0], meanshift_input[:, 1], c=labels, cmap=cmap, s=1)
            for j in range(len(centers)):
                plt.annotate("{}-{}".format(i+1, j+1), (centers[j, 0], centers[j, 1]))

    if timer:
        end_meanshift=time.time()
        print("MeanShift 用时:", str(datetime.timedelta(seconds=end_meanshift-start_meanshift)))
    
    if plot:
        plt.show()

    return midpoint_labels, segment_centers

def get_segments(roads, midpoints, midpoint_labels, plot=False, timer=True, write_file=True):
    """
    ```
    [
        [MultiLineString, MultiLineString, ...],
        [MultiLineString, MultiLineString, ...],
        ...
    ]
    ```
    """
    if timer:
        start_segment=time.time()

    if plot or write_file:
        segments_forplot=[]

    # len(roads_in)==len(roads)==len(points)==len(midpoints) 第一维都是有几条路
    segments=[[] for i in range(len(midpoints))]

    # midpoint 和 labels 一一对应，一个中点有一个标签
    # 例如 midpoints[labels==1] 可以取出 midpoints 里所有第 1 类的点
    for i in range(len(midpoints)):
        if not list(midpoint_labels[i]):
            continue

        k=len(np.unique(midpoint_labels[i]))
        for j in range(k):
            segment_points=np.array(midpoints[i])[midpoint_labels[i]==j]
            segment_points=[p["coord"] for p in segment_points]
            segment_points=MultiPoint(segment_points)
            segment_points=gp.GeoSeries(segment_points)
            convex_hull=segment_points.convex_hull.iloc[0]
            segment=roads.iloc[i].intersection(convex_hull)
            # XXX: 数据量过小的时候有可能会没有交集
            if segment.is_empty:
                continue
            segments[i].append(segment)
            if plot or write_file:
                segments_forplot.append(segment)

    if timer:
        end_segment=time.time()
        print("分割 segment 用时:", str(datetime.timedelta(seconds=end_segment-start_segment)))

    if plot or write_file:
        se=gp.GeoSeries(segments_forplot)
        show_geom(se, None, "segments", cmap=plt.cm.tab10, show=plot, write_file=write_file)

    if write_file:
        se.to_file(work_path+"supersegment_output/supersegment_result.shp", driver="ESRI Shapefile", encoding="utf-8")
        se.to_file(work_path+"supersegment_output/supersegment_result.json", driver="GeoJSON", encoding="utf-8")

    return segments

def supersegment(roads_in, buffer_distance, num_of_cars, cluster_method, plot=False, timer=True, write_file=True):
    """
    - Input:
        - roads_in: GeoSeries
    - Parameters: 
        - buffer_distance (recommend 0.00004)
        - num_of_cars
        - cluster_method ("kmeans" or "meanshift")
        - plot: bool, default=False
        - timer: bool, default=True
    - Returns:
        - cluster centers (3D array)
        - list of segments [[MultiLineString, MultiLineString, ...], [MultiLineString, MultiLineString, ...], ...]
    """

    # FIXME: 双向车道, 注意 buffer_distance 参数能否把两方向区分出来，但是太小的话有时候框不到点 -> 合并双向车道

    if cluster_method not in ("kmeans", "meanshift"):
        raise Exception("Invalid cluster method")

    if write_file:
        try:
            os.mkdir(work_path+"supersegment_output")
        except FileExistsError:
            pass

    if timer:
        start=time.time()

    roads=roads_in.apply(lambda x: x.buffer(distance=buffer_distance, cap_style=2))

    if plot:
        show_geom(gp.GeoDataFrame(geometry=roads), "blue", "road")

    # 1. GPS 点匹配到道路上
    if num_of_cars in (2000, 5000, 10000, 20000):
        tracks=get_tracks(file_path=work_path+"track_{}_cars.json".format(str(num_of_cars)), timer=timer)
    else:
        tracks=get_tracks(num_of_cars, timer=timer)

    points=get_matched_points(roads, tracks, plot=plot, timer=timer)

    # 2. 计算中点坐标和速度
    midpoints=get_midpoints(points, plot=plot, timer=timer, match=True, roads=roads)

    # 3. 将中点放入聚类模型
    if cluster_method=="kmeans":
        midpoint_labels, segment_centers=cluster_kmeans(midpoints, timer=timer, kmeans_details_plot=False, plot=plot)
    else:
        midpoint_labels, segment_centers=cluster_meanshift(midpoints, timer=timer, plot=plot)

    # 4. 凸包+交集求出 Line 类型的 segment
    segments=get_segments(roads_in, midpoints, midpoint_labels, timer=timer, plot=plot, write_file=write_file)

    if timer:
        end=time.time()
        print("总用时:", str(datetime.timedelta(seconds=end-start)))

    return segments, segment_centers

if __name__ == "__main__":
    methods=("kmeans", "meanshift")

    cluster_method=methods[0]

    df=read_boundary()

    buffer_distance=0.00004

    num_of_cars=10000

    # 羊市街+西玉龙街
    # roads=df.loc[(df["obj_id"]==283504) | (df["obj_id"]==283505) | (df["obj_id"]==283506), "geometry"]

    road_start_index=21
    road_end_index=None

    roads=df.loc[road_start_index:road_end_index, "geometry"]

    # (104.0421, 30.6526) (104.1287, 30.7271)
    print("Before drop: {} roads".format(len(roads)))

    drop_index=[]
    for i in roads.index:
        minx, miny, maxx, maxy=roads[i].bounds
        if maxx<104.0421 or maxy<30.6526 or minx>104.1287 or miny>30.7271:
            drop_index.append(i)
    roads=roads.drop(index=drop_index)
    
    print("After drop: {} roads".format(len(roads)))

    segments, segment_centers=supersegment(roads, buffer_distance, num_of_cars, cluster_method=cluster_method, timer=True, plot=True, write_file=False)

    print(np.array(segment_centers))

    # TODO: kmeans 和 meanshift 的函数有大量重复代码，重构
