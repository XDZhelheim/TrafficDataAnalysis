import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gp
from shapely.geometry import Polygon, MultiLineString, Point
import shapely.wkt as wkt
import supersegment

if __name__ == "__main__":
    df=pd.read_table("./TrafficDataAnalysis/boundary.txt")
    df['geometry']=df['geometry'].apply(wkt.loads)
    df=gp.GeoDataFrame(df)
    df.crs={'init':'epsg:4326'}

    # p=Point(104.05123456, 30.760912345)
    # poly: Polygon=df["geometry"][1693].buffer(0.0001)
    # print(poly.contains(p))
    # print(df.loc[(df["obj_id"]==281973) | (df["obj_id"]==281974)])

    # 使用 buffer+contains 实现了筛选此条路的 GPS 点
    roads=df.loc[(df["obj_id"]==283504) | (df["obj_id"]==283505) | (df["obj_id"]==283506), "geometry"].apply(lambda x: x.buffer(distance=0.0001))

    points=[]
    tracks=supersegment.get_tracks(1000)
    for track in tracks:
        for p in track:
            p=Point(p[0], p[1])
            for road in roads:
                if road.contains(p):
                    points.append(p)
    df_track=gp.GeoDataFrame(geometry=points)
    fig, ax=plt.subplots(figsize=(12, 8))
    ax.axis("off")
    df_track.plot(ax=ax, color="black", markersize=0.2)

    plt.show()
