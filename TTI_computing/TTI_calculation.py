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

def get_TTIspeed():
    return random.randint(15, 25)

def LL2Dist(Lat1, Lng1, Lat2, Lng2):
    """
    - 得到两个坐标点之间的距离，单位为 米
    """

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


def get_avg_speed(points, plot=False, timer=True):
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
            if (speed == 0) | (speed > 23):
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


def show_speed(avg_speeds):
    """
    - 显示当前的速度分布图，展示不同速度所占的比例
    """
    speed_list = []
    for each_list in avg_speeds:
        for speed in each_list:
            speed_list.append(speed[0])
    z_cnt = 0
    for i in speed_list:
        if i == 0:
            z_cnt += 1
    plt.figure()
    freq, bins, _ = plt.hist(speed_list, rwidth=0.8)
    plt.title('Velocity statistical analysis')
    plt.xlabel('speed (m/s)')
    plt.ylabel('amount of different speed')
    plt.show()


def show_TTIfig(avg_speed, TTI_interval, plot):
    """
    - 利用速度数据计算各时刻的TTI，将速度以及计算得到的TTI利用折线图的方式进行可视化
    """

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
    TTI = []
    free_speed_list = []
    for each_map in speed_map:
        lx = []
        ly_speed = []
        ly_tti = []
        ly_amount = []
        horizontal = []
        for x_time in each_map:
            speed_list = each_map[x_time]
            avg = np.mean(speed_list)
            lx.append(x_time / (60 / TTI_interval))
            ly_speed.append(avg)
            ly_amount.append(len(speed_list))
            horizontal.append(1)
        free_speed = max(ly_speed)
        free_speed_list.append(free_speed)
        for ever_speed in ly_speed:
            ly_tti.append(free_speed / ever_speed)
        TTI.append([lx, ly_tti])
        # 画出各个时刻的轨迹数量
        if plot:
            plt_figure('track distribution', 'Time (h)', 'track numbers', lx, ly_amount, False, False)
            plt_figure('Speed calculation', 'Time (h)', 'Speed (m/s)', lx, ly_speed, [0, 23],
                       [math.floor(min(ly_speed)), math.ceil(max(ly_speed))])
            plt_figure('TTI calculation', 'Time (h)', 'TTI', lx, ly_tti, [0, 23], False)
            # plt.figure()
            # plt.title('track distribution')
            # plt.xlabel('Time (h)')
            # plt.ylabel('track numbers')
            # plt.plot(lx, ly_amount)

            # 画出平均速度的折线图
            # plt.figure()
            # plt.xlim([0, 23])
            # # plt.ylim([math.floor(min(ly_speed)), math.ceil(max(ly_speed))])
            # plt.title('Speed calculation')
            # plt.xlabel('Time (h)')
            # plt.ylabel('Speed (m/s)')
            # plt.plot(lx, ly_speed, linewidth=3, color='r', marker='o')  # 画折线图 （平均速度）
            # plt.plot(lx, y2, linewidth=3, color='r', marker='o')  # 画折线图 （平均速度）
            # 画出TTI的折线图
            # plt.figure()
            # plt.xlim([0, 23])
            # # plt.ylim([0.9])
            # plt.title('TTI calculation')
            # plt.xlabel('Time (h)')
            # plt.ylabel('TTI')
            # plt.plot(lx, horizontal, label='second line', linewidth=1, color='b', linestyle='--')  # 画折线图 （平均速度）
            # plt.plot(lx, ly_tti, label='Frist line', linewidth=3, color='r', marker='o')  # 画折线图 （平均速度）
            # plt.plot(lx, y3, label='Frist line', linewidth=3, color='r', marker='o')  # 画折线图 （平均速度）
            plt.show()
    return TTI, free_speed_list


def plt_figure(title, xlabel, ylabel, x_list, y_list, x_lim, y_lim):
    """
    - 通过各参数，画出一张折线图
    """
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if x_lim:
        plt.xlim(x_lim)
    if y_lim:
        plt.ylim(y_lim)
    plt.plot(x_list, y_list)


def draw_road(roads):
    """
    - 在folium中，画出筛选过后的路段在地图上的显示
    """
    m = folium.Map(location=[30.20592, 103.21838])
    road_list = []
    for i in roads:
        # print(i)
        # print(type(i))
        # for j in i:
        #     print(j)
        road_list.append(i)
    folium.PolyLine(locations=roads, color='blue').add_to(m)
    m.save("1.html")
    import webbrowser
    webbrowser.open("1.html")


def cal_TTI(roads, buffer_distance, num_of_cars, plot, timer, TTI_interval):
    if timer:
        start = time.time()

    roads = roads.apply(lambda x: x.buffer(distance=buffer_distance, cap_style=2))
    if plot:
        show_geom(gp.GeoDataFrame(geometry=roads), "blue", "road")

    # 1. GPS 点匹配到道路上
    read1 = time.time()
    tracks = []
    for i in range(int(num_of_cars / 10000 + 1)):
        with open('E:\\didi\\城市交通指数和轨迹数据_2018\\data\\track_' + str(i * 10000) + '_cars', "rb") as f:
            tmp_read = pickle.load(f)
            tracks.extend(tmp_read)
    tracks = tracks[:num_of_cars]
    read2 = time.time()
    if timer:
        print('读取轨迹用时:', str(read2 - read1))
    points = get_matched_points(roads, tracks, plot=plot, timer=timer)
    # 2. 计算中点坐标和速度
    avg_speeeds = get_avg_speed(points, plot=plot, timer=timer)

    # 异常值处理  将 速度大于23的舍去
    if plot:
        show_speed(avg_speeeds)
    # 3. 按照时间将这些点分类，得到每个时间段的平均速度
    road_avg_speed = []
    for road_speed in avg_speeeds:
        road_avg_speed.append(np.mean([p[0] for p in road_speed]))

    TTI, free_speed = show_TTIfig(avg_speeeds, TTI_interval, plot)
    end = time.time()
    if timer:
        print("总共用时为：", str(end - start))

    return TTI, free_speed,road_avg_speed


if __name__ == "__main__":
    df = ss.read_boundary()
    buffer_distance = 0.00004
    num_of_cars = 2000
    TTI_interval = 30
    # 羊市街+西玉龙街
    roads = df.loc[(df["obj_id"] == 283506), "geom"]
    # roads = df.loc[(df["obj_id"] == 283505) | (df["obj_id"] == 283506) | (df["obj_id"] == 283524), "geom"]

    # roads = df.loc[(df["obj_id"] == 283505) | (df["obj_id"] == 283506), "geom"]

    TTI, free_speed,road_avg_speed = cal_TTI(roads, buffer_distance, num_of_cars, plot=False, timer=False, TTI_interval=TTI_interval)
