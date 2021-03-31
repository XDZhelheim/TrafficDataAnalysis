import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gp
from shapely.geometry import Point
import shapely.wkt as wkt
import time
import datetime
from pyproj import CRS
import json
import math
import random
import warnings
import pickle
import Supersegment.supersegment as ss
from scipy import interpolate
import folium

warnings.filterwarnings('ignore')

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


def read_boundary():
    df = pd.read_csv("E:/didi/城市交通指数和轨迹数据_2018/data/boundary.txt", sep='\t')
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
        start_read=time.time()

    if file_path:
        with open(file_path, "r") as f:
            tracks=json.load(f)
            tracks=list(tracks)

        return tracks

    df=pd.read_csv("E:/didi/城市交通指数和轨迹数据_2018/data/chengdushi_1001_1010.csv", nrows=num_of_cars, header=0, names=["track"], usecols=[2])
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


def LL2Dist(Lat1, Lng1, Lat2, Lng2):
    ra = 6378137.0  # radius of equator: meter
    rb = 6356752.3142451  # radius of polar: meter
    flatten = (ra - rb) / ra  # Partial rate of the earth
    if Lat1 == Lat2 and Lng1 == Lng2:
        return 0
    # change angle to radians
    radLatA = math.radians(Lat1)
    radLonA = math.radians(Lng1)
    radLatB = math.radians(Lat2)
    radLonB = math.radians(Lng2)

    pA = math.atan(rb / ra * math.tan(radLatA))
    pB = math.atan(rb / ra * math.tan(radLatB))
    x = math.acos(math.sin(pA) * math.sin(pB) + math.cos(pA) * math.cos(pB) * math.cos(radLonA - radLonB))
    c1 = (math.sin(x) - x) * (math.sin(pA) + math.sin(pB)) ** 2 / math.cos(x / 2) ** 2
    c2 = (math.sin(x) + x) * (math.sin(pA) - math.sin(pB)) ** 2 / math.sin(x / 2) ** 2
    dr = flatten / 8 * (c1 - c2)
    distance = ra * (x + dr)  ##单位:米
    return distance


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
    for n, track in enumerate(points):
        for i in range(len(track) - 1):
            if track[i]['track_id'] != track[i + 1]['track_id']:
                continue
            p1 = track[i]['coord']
            t1 = track[i]['time']
            p2 = track[i + 1]['coord']
            t2 = track[i + 1]['time']
            dis = LL2Dist(p1.x, p1.y, p2.x, p2.y)
            speed = dis / (t2 - t1)
            if (speed == 0 )|(speed > 23) :
                continue
            avg_speeds[n].append([speed, t1])

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


def show_Speed(avg_speeds):
    speed_list = []
    for each_list in avg_speeds:
        for speed in each_list:
            speed_list.append(speed[0])

    z_cnt = 0
    for i in speed_list:
        if i == 0:
            z_cnt += 1
    print(z_cnt)
    print(len(speed_list))
    plt.figure()
    # n = 6
    # bins = range(1,n+2)
    freq,bins,_ = plt.hist(speed_list,rwidth=0.8)
    plt.title('Velocity statistical analysis')
    plt.xlabel('speed (m/s)')
    plt.ylabel('amount of different speed')
    plt.show()
    for i in freq:
        print(i)
    return


def show_TTIfig(avg_speed, TTI_interval):
    speed_map = [{} for i in range(len(avg_speed))]
    for i in range(int(1440 / TTI_interval)):
        for x_time in speed_map:
            x_time[i] = []
    for n, each_roads in enumerate(avg_speed):
        for i in each_roads:
            tmp_time = i[1]
            format_time = time.localtime(tmp_time)
            minutes = format_time.tm_hour * 60 + format_time.tm_min
            speed_map[n][int(minutes / TTI_interval)].append(i[0])

    for each_map in (speed_map):
        lx = []
        ly_speed = []
        ly_tti = []
        ly_amount = []
        horizontal = []

        for x_time in each_map:
            speed_list = each_map[x_time]
            list_size = len(speed_list)
            random_list = random.sample(speed_list, int(list_size / 50))
            avg = np.mean(speed_list)
            lx.append(x_time / (60/TTI_interval))
            ly_speed.append(avg)
            ly_amount.append(len(speed_list))
            horizontal.append(1)
            # for x_time in speed_list:
            #     plt.scatter(x_time,avg, marker='.')  # 打散点图  （各个速度

        free_speed = max(ly_speed)
        for ever_speed in ly_speed:
            ly_tti.append(free_speed / ever_speed)

        xnew = np.arange(0, lx[-1], 0.01)
        # 实现函数
        func1 = interpolate.interp1d(lx, ly_amount, kind='cubic')
        func2 = interpolate.interp1d(lx, ly_speed, kind='cubic')
        func3 = interpolate.interp1d(lx, ly_tti, kind='cubic')
        # 利用xnew和func函数生成ynew,xnew数量等于ynew数量
        y1 = func1(xnew)
        y2 = func2(xnew)
        y3 = func3(xnew)
        plt.figure()
        plt.title("smooth amount")
        plt.plot(xnew,y1)
        plt.show()
        plt.figure()
        plt.title("smooth speed")
        plt.plot(xnew, y2)
        plt.show()
        plt.figure()
        plt.title("smooth tti")
        plt.plot(xnew, y3)
        plt.show()

        #画出各个时刻的轨迹数量
        plt.figure()
        plt.title('track distribution')
        plt.xlabel('Time (h)')
        plt.ylabel('track numbers')
        plt.plot(lx,ly_amount)
        # 画出平均速度的折线图
        plt.figure()
        plt.xlim([0,23])
        plt.ylim([math.floor(min(ly_speed)),math.ceil(max(ly_speed))])
        plt.title('Speed calculation')
        plt.xlabel('Time (h)')
        plt.ylabel('Speed (m/s)')
        plt.plot(lx, ly_speed, linewidth=3, color='r', marker='o')  # 画折线图 （平均速度）
        # plt.legend()

        # 画出TTI的折线图
        plt.figure()
        plt.xlim([0,23])
        # plt.ylim([0.9])
        plt.title('TTI calculation')
        plt.xlabel('Time (h)')
        plt.ylabel('TTI')
        plt.plot(lx, horizontal, label='second line', linewidth=1, color='b',linestyle='--')  # 画折线图 （平均速度）
        plt.plot(lx, ly_tti, label='Frist line', linewidth=3, color='r', marker='o')  #画折线图 （平均速度）
        plt.show()
    return


def draw_road(roads,points):
    m = folium.Map(location=[30.20592, 103.21838])

    folium.PolyLine(locations=roads, color='blue').add_to(m)

    m.save("1.html")
    import webbrowser
    webbrowser.open("1.html")
    exit(0)
    return


def cal_TTI(roads, buffer_distance, num_of_cars, cluster_method, plot=False, timer=True):
    if timer:
        start = time.time()

    roads = roads.apply(lambda x: x.buffer(distance=buffer_distance))

    if plot:
        show_geom(gp.GeoDataFrame(geometry=roads), "blue", "road")

    # 1. GPS 点匹配到道路上
    read1 = time.time()
    tracks = []
    for i in range(int(num_of_cars/10000 + 1)):
        with open('E:/didi/城市交通指数和轨迹数据_2018/data/track_' + str(i * 10000) + '_cars', "rb") as f:
            tmp_read = pickle.load(f)
            tracks.extend(tmp_read)

    tracks = tracks[:num_of_cars]
    read2 = time.time()
    print('read track used time')
    print(read2-read1)
    points = get_matched_points(roads, tracks, plot=plot, timer=timer)

    draw_road(roads,points)

    # 2. 计算中点坐标和速度
    avg_speeeds = get_avg_speed(points, plot=plot, timer=timer, match=True, roads=roads)


    # 异常值处理  将 速度大于23的舍去
    show_Speed(avg_speeeds)


    # 3. 按照时间将这些点分类，得到每个时间段的平均速度
    show_TTIfig(avg_speeeds, TTI_interval=60)


if __name__ == "__main__":
    methods = ("kmeans", "meanshift")

    df = read_boundary()

    buffer_distance = 0.00004

    num_of_cars = 100

    # 羊市街+西玉龙街
    # roads = df.loc[(df["obj_id"] == 283504) | (df["obj_id"] == 283505) | (df["obj_id"] == 283506), "geom"]
    roads = df.loc[(df["obj_id"] == 283506), "geom"]

    # road_start_index=21
    # road_end_index=22
    # roads=df.loc[road_start_index:road_end_index, "geometry"]

    cal_TTI(roads, buffer_distance, num_of_cars, cluster_method=methods[0], timer=True, plot=False)