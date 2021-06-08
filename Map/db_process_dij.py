import geopandas as gp
import networkx as nx
import shapely.wkt as wkt
from pyproj import CRS

graph=nx.read_shp("./boundary_shapefile/boundary.shp")

def get_length(edge):
    geom=wkt.loads(edge["Wkt"])

    roads=gp.GeoSeries(geom)
    roads.crs=CRS("epsg:4326")
    roads=roads.to_crs("epsg:2432")

    return roads.length[0]

import pandas as pd

df=pd.DataFrame(columns=["source_lng", "source_lat", "target_lng", "target_lat", "distance", "path"])

count=0
for node in graph.nodes:
    length_dict, path_dict=nx.single_source_dijkstra(graph, node, weight=lambda a, b, x: get_length(x))
    for key in length_dict:
        df.loc[len(df)]=[node[0], node[1], key[0], key[1], length_dict[key], path_dict[key]]

    count+=1
    if count%1000==0:
        df.to_csv("./db_data/dijkstra_{}.csv".format(count/1000), index=False, header=0)
        df=pd.DataFrame(columns=["source_lng", "source_lat", "target_lng", "target_lat", "distance", "path"])

df.to_csv("./db_data/dijkstra_last.csv", index=False, header=0)
