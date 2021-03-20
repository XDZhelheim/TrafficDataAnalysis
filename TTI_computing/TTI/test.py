# # # -*- coding: utf-8 -*-
# #
# # import matplotlib.pyplot as plt
# #
# # path = "E:\didi\城市交通指数\成都市\成都市.txt"
# # city = 'E:/didi/城市交通指数/成都市/city_district.txt'
# # boundary = 'E:\didi\城市交通指数\成都市\boundary.txt'
# #
# # # city_list = []
# # chengdu_list = []
# # with open(city) as f:
# #     # print(f)
# #     city_list = f.read().split('\n')
# #
# #     for i in city_list:
# #         i = i.split()
# #         if len(i) == 0:
# #             continue
# #         if i[0] == '17':
# #             chengdu_list.append(i)
# # TTI = []
# # speed = []
# # time = []
# # chengdu_list = chengdu_list[0:144]
# # # chengdu_list = chengdu_list[0:300]
# # for n, i in enumerate(chengdu_list):
# #     TTI.append(float(i[3]))
# #     speed.append(float(i[4]))
# #     time.append(i[2][:5])
# #     # time.append(n)
# #     # print(i[2])
# #
# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# #
# # lns1 = ax.plot(time, TTI, '-g', label='T T I')
# # ax2 = ax.twinx()
# # lns2 = ax2.plot(time, speed, '-r', label='speed')
# #
# # # added these three lines
# # lns = lns1 + lns2
# # labs = [l.get_label() for l in lns]
# # ax.legend(lns, labs, loc=0)
# #
# # ax.grid()
# # ax.set_xlabel("Time (h)")
# # ax.set_ylabel("T T I")
# # # ax2.set_ylabel(r"speed ($^\circ$C)")
# # ax2.set_ylabel("speed (km/h)")
# # x_major_locator = plt.MultipleLocator(18)
# # ax.xaxis.set_major_locator(x_major_locator)
# #
# # plt.title('成都市TTI', fontproperties="SimHei")
# # plt.savefig('成都.png')
# # plt.show()
# #
#
# # import time
# # timeArray = time.localtime(1538323676)
# # # timeArray = time.strftime("%Y-%m-%d %H:%M:%S",timeArray)
# # timeArray = int(time.strftime("%H",timeArray))
# # print(timeArray)
#
# import math
#
# def LL2Dist(Lat1,Lng1,Lat2,Lng2):
#     ra = 6378137.0        # radius of equator: meter
#     rb = 6356752.3142451  # radius of polar: meter
#     flatten = (ra - rb) / ra  # Partial rate of the earth
#     if Lat1==Lat2 and Lng1==Lng2:
#         return 0
#     # change angle to radians
#     radLatA = math.radians(Lat1)
#     radLonA = math.radians(Lng1)
#     radLatB = math.radians(Lat2)
#     radLonB = math.radians(Lng2)
#
#     pA = math.atan(rb / ra * math.tan(radLatA))
#     pB = math.atan(rb / ra * math.tan(radLatB))
#     x = math.acos(math.sin(pA) * math.sin(pB) + math.cos(pA) * math.cos(pB) * math.cos(radLonA - radLonB))
#     c1 = (math.sin(x) - x) * (math.sin(pA) + math.sin(pB))**2 / math.cos(x / 2)**2
#     c2 = (math.sin(x) + x) * (math.sin(pA) - math.sin(pB))**2 / math.sin(x / 2)**2
#     dr = flatten / 8 * (c1 - c2)
#     distance = ra * (x + dr)  ##单位:米
#     return distance
#
# if __name__ == '__main__':
#     print(LL2Dist(104.04949 ,30.70615 ,104.04994, 30.70587 ))
#     print(LL2Dist(30.70615 ,104.04949 , 30.70587, 104.04994))

import random
# a = [1,2,5,23,53,13,535]
a = [i for i in range(42)]
b = random.sample(a,10)
print(b)