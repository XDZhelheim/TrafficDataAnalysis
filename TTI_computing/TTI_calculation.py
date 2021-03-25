import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gp
from shapely.geometry import Point
import shapely.wkt as wkt
from sklearn.cluster import KMeans, MeanShift
import time
import datetime
from pyproj import CRS
import json
import math

colors = ["#A1E2E6", "#E6BDA1", "#B3A16B", "#678072", "#524A4A"]


def show_district_and_road(df, district_line_number_start, district_line_number_end, road_line_number_start,
                           road_line_number_end):
    """
    - 此方法用来画城市和道路的整体图
    - boundary.txt 的 0~20 行是区域的边界信息, 21 及之后是道路
    - Typical use case: show_district_and_road(df, None, 20, 21, None) 注意是闭区间
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    df.loc[district_line_number_start:district_line_number_end].plot(ax=ax, color=colors[1], edgecolor="white",
                                                                     linewidth=0.3)
    plt.title("成都市", fontsize=30, fontname="Source Han Serif CN", color="black")
    ax.axis("off")
    fig.canvas.set_window_title(u'成都市')

    df.loc[road_line_number_start:road_line_number_end].plot(ax=ax, color=colors[4], linewidth=0.2)

    plt.show()


def show_geom(df, color, title):
    """
    - 画几何图形
    - df 为 geodataframe
    - 使用例:
        - df_midpoint=gp.GeoDataFrame(geometry=midpointcoords)
        - show_points(df_midpoint, "red", "中点")
    """

    fig, ax = plt.subplots(figsize=(12, 8))
    df.plot(ax=ax, color=color, markersize=0.2)
    plt.title(title, fontsize=15, fontname="Source Han Serif CN", color="black")
    ax.axis("off")
    fig.canvas.set_window_title(title)
    plt.show()


def kmeans(kmeans_input, k=None, plot=False):
    # TODO: 选择 k 的算法
    if not k:
        # 利用SSE选择k, 手肘法
        SSE = []  # 存放每次结果的误差平方和
        for k in range(1, 9):
            estimator = KMeans(n_clusters=k)
            estimator.fit(kmeans_input)
            SSE.append(estimator.inertia_)

        diff = []
        for i in range(len(SSE) - 1):
            diff.append(SSE[i] - SSE[i + 1])

        for i in range(len(diff)):
            if diff[i] <= diff[0] / 20:
                k = i + 1
                break

        if plot:
            X = range(1, 9)
            plt.subplot(221)
            plt.xlabel('k')
            plt.ylabel('SSE')
            plt.plot(X, SSE, 'o-')

    km_model = KMeans(n_clusters=k)
    km_model.fit(kmeans_input)

    # 分类信息
    labels = km_model.labels_
    centers = km_model.cluster_centers_

    if plot:
        # 3d所有点
        ax = plt.subplot(222, projection='3d')
        ax.scatter(kmeans_input[:, 0], kmeans_input[:, 1], kmeans_input[:, 2], c=labels, cmap=plt.cm.tab10)

        # 2d所有点
        plt.subplot(223)
        plt.scatter(kmeans_input[:, 0], kmeans_input[:, 1], c=labels, cmap=plt.cm.tab10, s=30)
        for i in range(k):
            plt.annotate(str(i + 1), (centers[i, 0], centers[i, 1]))

        # 3d 中心点 s为大小
        ax = plt.subplot(224, projection='3d')
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='*', s=80)

        # ax.axis("off")
        plt.show()

    return k, labels, centers


def read_boundary():
    df = pd.read_csv("./data/boundary.txt",sep='\t')
    df['geom'] = df['geom'].apply(wkt.loads)
    df = gp.GeoDataFrame(df)
    df.crs = CRS("epsg:4326")

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
        start_read = time.time()

    if file_path:
        with open(file_path, "r") as f:
            tracks = json.load(f)
            tracks = list(tracks)

        return tracks

    df = pd.read_csv("./data/chengdushi_1001_1010.csv", nrows=num_of_cars, header=0, names=["track"],
                     usecols=[2])
    tracks = []
    for temp in df["track"]:
        temp = temp.lstrip("[").rstrip("]")
        temp = temp.split(", ")  # 注意分隔符是逗号+空格
        for i in range(len(temp)):
            temp[i] = temp[i].split(" ")
        for item in temp:
            item[0] = float(item[0])
            item[1] = float(item[1])
            item[2] = int(item[2])
        tracks.append(temp)

    if timer:
        end_read = time.time()
        print("读轨迹用时:", str(datetime.timedelta(seconds=end_read - start_read)))

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
        points_forplot = []  # 只用来画图

    if timer:
        start_gps = time.time()

    bound_list = []
    for road in roads:
        bound_list.append(road.bounds)  # road.bounds=(minx, miny, maxx, maxy)

    minx = 999
    miny = 999
    maxx = 0
    maxy = 0
    for bound in bound_list:
        if bound[0] < minx:
            minx = bound[0]
        if bound[1] < miny:
            miny = bound[1]
        if bound[2] > maxx:
            maxx = bound[2]
        if bound[3] > maxy:
            maxy = bound[3]

    points = [[[] for i in range(len(tracks))] for j in range(len(roads))]
    for i in range(len(tracks)):
        for gps in tracks[i]:

            # 框点，效果拔群
            if gps[0] < minx or gps[0] < maxy or gps[1] < miny or gps[1] > maxy:
                continue

            p = Point(gps[0], gps[1])
            for j in range(len(roads)):
                road = roads.iloc[j]
                if road.contains(p):
                    points[j][i].append({"coord": p, "time": gps[2]})
                    if plot:
                        points_forplot.append(p)
                    break

    if timer:
        end_gps = time.time()
        print("GPS 点匹配用时:", str(datetime.timedelta(seconds=end_gps - start_gps)))

    # 画所有点
    if plot:
        df_track = gp.GeoDataFrame(geometry=points_forplot)
        show_geom(df_track, "black", "所有点")

    return points


def get_matched_points2(roads, tracks, plot=False, timer=True):
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
        points_forplot = []  # 只用来画图

    if timer:
        start_gps = time.time()

    bound_list = []
    for road in roads:
        bound_list.append(road.bounds)  # road.bounds=(minx, miny, maxx, maxy)

    minx = 999
    miny = 999
    maxx = 0
    maxy = 0
    for bound in bound_list:
        if bound[0] < minx:
            minx = bound[0]
        if bound[1] < miny:
            miny = bound[1]
        if bound[2] > maxx:
            maxx = bound[2]
        if bound[3] > maxy:
            maxy = bound[3]

    # points = [[[] for i in range(len(tracks))] for j in range(len(roads))]
    points = [[] for i in range(len(roads))]
    for i in range(len(tracks)):
        for gps in tracks[i]:
            # 框点，效果拔群
            if gps[0] < minx or gps[0] < maxy or gps[1] < miny or gps[1] > maxy:
                continue
            p = Point(gps[0], gps[1])
            for j in range(len(roads)):
                road = roads.iloc[j]
                if road.contains(p):
                    points[j].append({"coord": p, "time": gps[2], "track_id": i})
                    if plot:
                        points_forplot.append(p)
                    break

    if timer:
        end_gps = time.time()
        print("GPS 点匹配用时:", str(datetime.timedelta(seconds=end_gps - start_gps)))

    # 画所有点
    if plot:
        df_track = gp.GeoDataFrame(geometry=points_forplot)
        show_geom(df_track, "black", "所有点")

    return points


def get_midpoints(points, match=True, roads=None, plot=False, timer=True):
    """
    - match 表示是否只保留在路上的中点
    - 如果一条路是弧形，算出的中点可能不在路上
    """
    if plot:
        midpoints_forplot = []

    if timer:
        start_mid = time.time()

    midpoints = [[] for i in range(len(points))]
    for i in range(len(points)):
        for track in points[i]:
            for j in range(len(track) - 1):
                midpoint = {}
                midpoint["coord"] = Point((track[j]["coord"].x + track[j + 1]["coord"].x) / 2,
                                          (track[j]["coord"].y + track[j + 1]["coord"].y) / 2)

                if match:
                    if not roads.iloc[i].contains(midpoint["coord"]):
                        continue

                # 一条路上多辆车同时出现
                # if track[j]["time"]==track[j+1]["time"]:
                #     continue

                midpoint["speed"] = track[j]["coord"].distance(track[j + 1]["coord"]) / abs(
                    track[j + 1]["time"] - track[j]["time"])
                midpoints[i].append(midpoint)
                if plot:
                    midpoints_forplot.append(midpoint)

    if timer:
        end_mid = time.time()
        print("计算中点用时:", str(datetime.timedelta(seconds=end_mid - start_mid)))

    if plot:
        # 画中点
        midpointcoords = []
        for midpoint in midpoints_forplot:
            midpointcoords.append(midpoint["coord"])
        df_midpoint = gp.GeoDataFrame(geometry=midpointcoords)
        show_geom(df_midpoint, "red", "中点")

    return midpoints


def LL2Dist(Lat1,Lng1,Lat2,Lng2):
    ra = 6378137.0        # radius of equator: meter
    rb = 6356752.3142451  # radius of polar: meter
    flatten = (ra - rb) / ra  # Partial rate of the earth
    if Lat1==Lat2 and Lng1==Lng2:
        return 0
    # change angle to radians
    radLatA = math.radians(Lat1)
    radLonA = math.radians(Lng1)
    radLatB = math.radians(Lat2)
    radLonB = math.radians(Lng2)

    pA = math.atan(rb / ra * math.tan(radLatA))
    pB = math.atan(rb / ra * math.tan(radLatB))
    x = math.acos(math.sin(pA) * math.sin(pB) + math.cos(pA) * math.cos(pB) * math.cos(radLonA - radLonB))
    c1 = (math.sin(x) - x) * (math.sin(pA) + math.sin(pB))**2 / math.cos(x / 2)**2
    c2 = (math.sin(x) + x) * (math.sin(pA) - math.sin(pB))**2 / math.sin(x / 2)**2
    dr = flatten / 8 * (c1 - c2)
    distance = ra * (x + dr)  ##单位:米
    return distance


def distance(p1, p2):
    return LL2Dist(p1[1],p1[0],p2[1],p2[0])


def get_avg_speed(points, match=True, roads=None, plot=False, timer=True):
    """
    - match 表示是否只保留在路上的中点
    - 如果一条路是弧形，算出的中点可能不在路上
    """
    if plot:
        midpoints_forplot = []

    if timer:
        start_mid = time.time()

    avg_speeds = [[] for i in range(len(points))]
    for track in points:
        for i in range(len(track)-1):
            if track[i]['track_id'] != track[i+1]['track_id']:
                continue
            p1 = track[i]['coord']
            t1 = track[i]['time']
            p2 = track[i+1]['coord']
            t2 = track[i+1]['time']
            dis = LL2Dist(p1.x,p1.y,p2.x,p2.y)
            speed = dis / (t2-t1)
            if speed == 0 | speed >40:
                continue
            avg_speeds.append([speed,t1])

    if timer:
        end_mid = time.time()
        print("计算平均速度用时:", str(datetime.timedelta(seconds=end_mid - start_mid)))

    if plot:
        # 画中点
        midpointcoords = []
        for midpoint in midpoints_forplot:
            midpointcoords.append(midpoint["coord"])
        df_midpoint = gp.GeoDataFrame(geometry=midpointcoords)
        show_geom(df_midpoint, "red", "中点")

    return avg_speeds


def get_segment_centers_kmeans(midpoints, k=None, kmeans_details_plot=False, timer=True):
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
        start_kmeans = time.time()

    segment_centers = []

    for i in range(len(midpoints)):
        if not midpoints[i]:  # 没有车经过这条路
            continue

        kmeans_input = []
        for midpoint in midpoints[i]:
            kmeans_input.append([midpoint["coord"].x, midpoint["coord"].y, midpoint["speed"]])
        kmeans_input = np.array(kmeans_input)

        k, labels, centers = kmeans(kmeans_input, k, kmeans_details_plot)

        segment_centers.append(centers)

        # 调整不同配色
        if i % 2:
            cmap = plt.cm.tab10
        else:
            cmap = plt.cm.Paired

        plt.scatter(kmeans_input[:, 0], kmeans_input[:, 1], c=labels, cmap=cmap, s=1)
        for j in range(k):
            plt.annotate("{}-{}".format(i + 1, j + 1), (centers[j, 0], centers[j, 1]))

    if timer:
        end_kmeans = time.time()
        print("KMeans 用时:", str(datetime.timedelta(seconds=end_kmeans - start_kmeans)))

    plt.show()

    return segment_centers


def get_segment_centers_meanshift(midpoints, timer=True):
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
        start_meanshift = time.time()

    segment_centers = []

    ms_model = MeanShift(n_jobs=-1)

    for i in range(len(midpoints)):
        if not midpoints[i]:  # 没有车经过这条路
            continue

        meanshift_input = []
        for midpoint in midpoints[i]:
            meanshift_input.append([midpoint["coord"].x, midpoint["coord"].y, midpoint["speed"]])
        meanshift_input = np.array(meanshift_input)

        ms_model.fit(meanshift_input)

        labels = ms_model.labels_
        centers = ms_model.cluster_centers_

        segment_centers.append(centers)

        # 调整不同配色
        if i % 2:
            cmap = plt.cm.tab10
        else:
            cmap = plt.cm.Paired

        plt.scatter(meanshift_input[:, 0], meanshift_input[:, 1], c=labels, cmap=cmap, s=1)
        for j in range(len(centers)):
            plt.annotate("{}-{}".format(i + 1, j + 1), (centers[j, 0], centers[j, 1]))

    if timer:
        end_meanshift = time.time()
        print("MeanShift 用时:", str(datetime.timedelta(seconds=end_meanshift - start_meanshift)))

    plt.show()

    return segment_centers


def cal_TTI(roads, buffer_distance, num_of_cars, cluster_method, plot=False, timer=True):
    if timer:
        start = time.time()

    roads = roads.apply(lambda x: x.buffer(distance=buffer_distance))

    if plot:
        show_geom(gp.GeoDataFrame(geometry=roads), "blue", "road")

    # 1. GPS 点匹配到道路上
    if num_of_cars in (2000, 5000, 10000, 20000):
        tracks = get_tracks(file_path="./data/track_" + str(num_of_cars) + "_cars.json", timer=timer)
    else:
        tracks = get_tracks(num_of_cars, timer=timer)

    points = get_matched_points2(roads, tracks, plot=plot, timer=timer)

    # 2. 计算中点坐标和速度
    midpoints = get_avg_speed(points, plot=plot, timer=timer, match=True, roads=roads)
    print(midpoints)


def supersegment(roads, buffer_distance, num_of_cars, cluster_method, plot=False, timer=True):
    """
    - Input:
        - roads: geometry (MULTILINESTRING)
    - Parameters: 
        - buffer_distance
        - num_of_cars
        - cluster_method ("kmeans" or "meanshift")
    - Output:
        - cluster centers
        - list of segments [Line, Line, Line, ...]
    """

    # FIXME: 双向车道, 注意 buffer_distance 参数能否把两方向区分出来，但是太小的话有时候框不到点 -> 合并双向车道
    # TODO: 分割 line

    if cluster_method not in ("kmeans", "meanshift"):
        raise Exception("Invalid cluster method")

    if timer:
        start = time.time()

    roads = roads.apply(lambda x: x.buffer(distance=buffer_distance))

    if plot:
        show_geom(gp.GeoDataFrame(geometry=roads), "blue", "road")

    # 1. GPS 点匹配到道路上
    if num_of_cars in (2000, 5000, 10000, 20000):
        tracks = get_tracks(file_path="./data/track_" + str(num_of_cars) + "_cars.json", timer=timer)
    else:
        tracks = get_tracks(num_of_cars, timer=timer)

    points = get_matched_points(roads, tracks, plot=plot, timer=timer)

    # 2. 计算中点坐标和速度
    midpoints = get_midpoints(points, plot=plot, timer=timer, match=True, roads=roads)

    # 3. 将中点放入聚类模型
    if cluster_method == "kmeans":
        segment_centers = get_segment_centers_kmeans(midpoints, timer=timer, kmeans_details_plot=False)
    else:
        segment_centers = get_segment_centers_meanshift(midpoints, timer=timer)

    if timer:
        end = time.time()
        print("总用时:", str(datetime.timedelta(seconds=end - start)))

    print(np.array(segment_centers))


if __name__ == "__main__":
    methods = ("kmeans", "meanshift")

    df = read_boundary()

    buffer_distance = 0.00004

    num_of_cars = 1000

    # 羊市街+西玉龙街
    roads = df.loc[(df["obj_id"] == 283504) | (df["obj_id"] == 283505) | (df["obj_id"] == 283506), "geom"]

    # road_start_index=21
    # road_end_index=22

    # roads=df.loc[road_start_index:road_end_index, "geometry"]

    cal_TTI(roads, buffer_distance, num_of_cars, cluster_method=methods[0], timer=True, plot=False)
