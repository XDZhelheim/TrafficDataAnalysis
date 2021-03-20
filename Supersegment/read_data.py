import pandas as pd
import numpy as np
from RegionFactory import RegionFactory
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()
import random

class District:
    def __init__(self, obj_id, obj_name, boundary):
        self.obj_id=obj_id
        self.obj_name=obj_name
        self.boundary=boundary

if __name__ == "__main__":
    df=pd.read_table("./TrafficDataAnalysis/boundary.txt", nrows=10000)
    # obj_id=281863 之后是路
    df['tti']=[0]*len(df)
    df['speed']=[0]*len(df)
    # print(df.loc[df['obj_id']==841])
    # 2018-1-1: 2736
    df2=pd.read_table("./TrafficDataAnalysis/city_district.txt", nrows=2736)
    for index, row in df2.iterrows():
        df.loc[df['obj_id']==row['obj_id'], 'tti']=row['tti']
        df.loc[df['obj_id']==row['obj_id'], 'speed']=row['speed']
    print(df)

    factory=RegionFactory().get_instance()
    region_coords=[]
    # reg=factory.create_region(0, 0, df[df.obj_id==281863]["geometry"][df[df.obj_id==281863].index[0]])
    for index, row in df.iterrows():
        region=factory.create_region(int(row['obj_id']), row['obj_name'], row['geometry'], float(row['tti']), float(row['speed']))
        if region:
            x, y=region.getCenter()
            speed=random.uniform(10, 100)
            region_coords.append([x, y, speed])

    region_coords=np.array(region_coords)

    # 利用SSE选择k, 手肘法
    SSE = []  # 存放每次结果的误差平方和
    for k in range(1, 9):
        estimator = KMeans(n_clusters=k)
        estimator.fit(region_coords)
        SSE.append(estimator.inertia_)
    X = range(1, 9)
    plt.subplot(221)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.plot(X, SSE, 'o-')

    k=3
    km_model = KMeans(n_clusters=k)
    km_model.fit(region_coords)

    # 分类信息
    y_pred = km_model.predict(region_coords)

    centers = km_model.cluster_centers_
    print(centers)

    # 3d所有点
    ax=plt.subplot(222, projection = '3d')
    ax.scatter(region_coords[:, 0], region_coords[:, 1], region_coords[:, 2], c=y_pred, cmap=plt.cm.tab10)
    # print(type(region_coords[:, 2][0]))

    # 2d所有点
    plt.subplot(223)
    plt.scatter(region_coords[:, 0], region_coords[:, 1], c=y_pred, cmap=plt.cm.tab10)
    # for i in range(k):
    #     plt.annotate(str(i+1), (centers[i, 0], centers[i, 1]))

    # 3d 中心点 s为大小
    ax=plt.subplot(224, projection = '3d')
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='*', s=80)

    plt.show()
    