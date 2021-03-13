import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gp
from shapely.geometry import Polygon, MultiLineString, Point
import shapely.wkt as wkt
import supersegment

# # Fixing random state for reproducibility
# np.random.seed(19680801)


# def randrange(n, vmin, vmax):
#     '''
#     Helper function to make an array of random numbers having shape (n, )
#     with each number distributed Uniform(vmin, vmax).
#     '''
#     return (vmax - vmin)*np.random.rand(n) + vmin

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# n = 100

# # For each set of style and range settings, plot n random points in the box
# # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
#     xs = randrange(n, 23, 32)
#     ys = randrange(n, 0, 100)
#     zs = randrange(n, zlow, zhigh)
#     ax.scatter(xs, ys, zs, marker=m)

# print(type(zs[0]))

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()


# df=pd.read_table("./TrafficDataAnalysis/boundary.txt", nrows=10000)
# df['tti']=[0]*len(df)
# df['speed']=[0]*len(df)
# # print(df.loc[df['obj_id']==841])

# # 2018-1-1
# df2=pd.read_table("./TrafficDataAnalysis/city_district.txt", nrows=2736)
# # print(df2)
# for index, row in df2.iterrows():
#     df.loc[df['obj_id']==row['obj_id'], 'tti']=row['tti']
#     df.loc[df['obj_id']==row['obj_id'], 'speed']=row['speed']
# print(df)

# df1=pd.read_table("./TrafficDataAnalysis/res2.txt", header=None, sep=' ')
# print(df1[0])
# x = np.linspace(200, 3500, 2000)
# plt.plot(x, x, '-r')
# # plt.plot(x, 0.15*x+750, '-r')
# plt.scatter(df1[0], df1[1])
# plt.show()

# df=pd.read_csv("./TrafficDataAnalysis/chengdushi_1001_1010.csv", nrows=1, header=0, names=["track"], usecols=[2])
# track=[]
# for temp in df["track"]:
#     temp=temp.lstrip("[").rstrip("]")
#     # print(temp)
#     # temp=temp.replace(", ", ";")
#     temp=temp.split(", ")
#     for i in range(len(temp)):
#         temp[i]=temp[i].split(" ")
#     for item in temp:
#         item[0]=float(item[0])
#         item[1]=float(item[1])
#         item[2]=int(item[2])
#     track.append(temp)
# print(track)

# with open("./TrafficDataAnalysis/chengdushi_1001_1010.csv") as f:
#     temp=f.readline()
# print(temp)

# df=pd.read_table("./TrafficDataAnalysis/boundary.txt", nrows=10000)
# df['geometry']=df['geometry'].apply(lambda z: wkt.loads(z))
# df=gp.GeoDataFrame(df)
# df.crs={'init':'epsg:4326'}

# supersegment.show_district_and_road(df, None, 21, 21, None)

# roads=df.loc[(df["obj_id"]==283504) | (df["obj_id"]==283505) | (df["obj_id"]==283506), "geometry"].apply(lambda x: x.buffer(distance=0.0001))
# print(roads.iloc[0])

p1=Point(1, 2)
p2=Point(1, 2)
d1={"coord": p1, "time": 10}
d2={"coord": p2, "time": 20}
print(d1, d2)