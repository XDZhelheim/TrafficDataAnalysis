import pandas as pd
import numpy as np
import math
import folium
import time
import matplotlib.pyplot as plt
import random

m = folium.Map(location=[30.20592,103.21838])
# folium.Circle(radius=float(i[2]), location=(float(i[0]), float(i[1])), popup='The Waterfront', color='crimson',
#               fill=False).add_to(m)
# folium.Circle(radius=i[3], location=(float(i[0]), float(i[1])), popup='The Waterfront', color='green',
#               fill=False).add_to(m)
# ls = folium.PolyLine(locations=line_arr, color='blue')

road = "../data/boundary.txt"
city = "../data/city_district.txt"
# path = "../data/top100.csv"
path = "../../data/chengdushi_1001_1010.csv"

time_flag = 0
posi_flag = 0
start_time = 1  # 这两个时间是，在进行计算时，要选取的时间段
end_time = 2


# 这个算法,几百km误差几十米,几千km误差几百米，不知道叫什么算法
# 两个经纬度之间的距离,椭球
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


def get_road_range(id):
    # open_road = pd.read_table(road)
    open_road = pd.read_csv(road, sep='\t')
    # id = 70
    print(open_road['obj_name'][id] + "  name")
    r_range = open_road['geom'][id]

    r_range = r_range[17:-2].split('),(')
    for i in r_range:
        t = i.split(",")
        for j in t:
            ans = j.split(' ')
            # folium.Marker([float(ans[1]),float(ans[0])]).add_to(m)
            # folium.Circle(radius=float(i[2]), location=(float(i[0]), float(i[1])), popup='The Waterfront', color='crimson',
            #               fill=False).add_to(m)
        # print(t)
    return r_range


# def get_track(start_time, end_time, road_range):
def get_track(num_of_cars):
    open_path = pd.read_csv(path, nrows=num_of_cars)
    # open_path = pd.read_csv(path)
    gps_record = open_path['gps'].values

    for each_record in gps_record:
        record = each_record[1:-1]
        record_list = record.split(',')
        print(record)
        for i in record_list:
            i_list = i.split(' ')
            longitude = i_list[0]
            latitude = i_list[1]
            tmp_time = i_list[2]


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
    # df = pd.read_csv("../data/chengdushi_1001_1010.csv", nrows=num_of_cars, header=0, names=["track"], usecols=[2])
    df = pd.read_csv("../../data/chengdushi_1001_1010.csv", nrows=num_of_cars, names=["track"], usecols=[2])
    track = []
    order_num = 0
    for temp in df["track"]:
        temp = temp.lstrip("[").rstrip("]")
        temp = temp.split(", ")  # 注意分隔符是逗号+空格
        for i in range(len(temp)):
            temp[i] = temp[i].split(" ")
        for item in temp:
            item[0] = float(item[0])
            item[1] = float(item[1])
            item[2] = int(item[2])
            item.append(order_num)
            # item[3] = order_num
        track.append(temp)
        order_num += 1
    return track


def distance(p1, p2):
    return LL2Dist(p1[1],p1[0],p2[1],p2[0])


def pos_onroad(posi, road_range):
    gps_pos = [float(posi[0]), float(posi[1])]
    for each_road in road_range:
        road_list = each_road.split(',')
        for each_road_point in road_list:
            point_list = each_road_point.split(' ')
            gps_road = [float(point_list[0]), float(point_list[1])]
            if distance(gps_road, gps_pos) < 20:
                return True
    return False


if __name__ == '__main__':
    time_interval = 100
    road_range = get_road_range(47)
    # for i in range(5):
    #     get_road_range(i+45)

    num_of_cars = 100  # 最后调整到了 20000
    # 1. 筛选此条路的 GPS 点
    tracks = get_tracks(num_of_cars)
    points = []
    cont = 0
    for i in road_range:
        t = i.split(' ')
        # folium.Marker([float(t[1]),float(t[0])]).add_to(m)
        # folium.Marker([float(t[1]),float(t[0])])

    for each_track in tracks:
        for point in each_track:
            cont += 1
            # if pos_onroad(point, road_range):
            # if 104.06060 <= point[0] <= 104.07049 and 30.66680 <= point[1] <= 30.66718:
            if True:
                points.append(point)
                folium.Circle(radius=10, location=(float(point[1]), float(point[0])), popup='The Waterfront', color='crimson',
                              fill=False).add_to(m)

    m.save("1.html")
    import webbrowser
    webbrowser.open("1.html")
    exit(0)

    # 2. 计算速度

    avg_speed = []
    for i in range(len(points) - 1):
        if points[i + 1][3] == points[i][3]:
            p1 = [points[i][0], points[i][1]]
            t1 = points[i][2]
            p2 = [points[i + 1][0], points[i + 1][1]]
            t2 = points[i+1][2]
            dis = distance(p1, p2)

            speed = distance(p1, p2) / (t2-t1)
            # print(speed)
            if speed == 0:
                continue
                # print(points[i])
                # print(points[i+1])
                # print('----------')
            if speed > 30:
                continue
            avg_speed.append([speed, points[i][2]])
    # 3. 按照时间将这些点分类，得到每个时间段的平均速度

    fig = plt.figure()
    speed_map = {}
    for i in range(24):
        speed_map[i] = []

    for i in avg_speed:
        tmp_time = i[1]
        # timeArray = (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(tmp_time)))
        timeArray = int(time.strftime("%H", time.localtime(tmp_time)))
        speed_map[timeArray].append(i[0])

        # print(timeArray)
    lx = []
    ly = []
    l_tti = []
    free_speed = np.mean(speed_map[3])
    for i in speed_map:
        speed_list = speed_map[i]
        list_size = len(speed_list)
        random_list = random.sample(speed_list,int(list_size/50))

        avg = np.mean(speed_list)
        lx.append(i)
        ly.append(avg)
        l_tti.append(free_speed/avg)
        for j in speed_list:
            plt.scatter(i,j,marker='.')  # 打散点图  （各个速度）

    # plt.plot(lx, ly, label='Frist line', linewidth=3, color='r', marker='o')  #画折线图 （平均速度）
    # plt.plot(lx, l_tti, label='Frist line', linewidth=3, color='r', marker='o')  #画折线图 （平均速度）
    plt.show()

    # m.save("1.html")
    # import webbrowser
    # webbrowser.open("1.html")
    # exit(0)

# folium.Circle(radius=float(i[2]), location=(float(i[0]), float(i[1])), popup='The Waterfront', color='crimson',
#               fill=False).add_to(m)
# folium.Circle(radius=i[3], location=(float(i[0]), float(i[1])), popup='The Waterfront', color='green',
#               fill=False).add_to(m)
# ls = folium.PolyLine(locations=line_arr, color='blue')


# folium.Circle(location=(float(ans[1]), float(ans[0])), radius=1).add_to(m)
