import json
import networkx as nx
import pandas as pd
import geopandas as gp
import shapely.wkt as wkt
import matplotlib.pyplot as plt
from pyproj import CRS

work_path="D:\\Codes\\PythonWorkspace\\TrafficDataAnalysis\\"

def read_boundary():
    df=pd.read_table(work_path+"boundary.txt")
    df['geometry']=df['geometry'].apply(wkt.loads)
    df=gp.GeoDataFrame(df)

    return df

def write_boundary_shp():
    df=read_boundary()
    roads=df.loc[21:, "geometry"]

    drop_index=[]
    for i in roads.index:
        minx, miny, maxx, maxy=roads[i].bounds
        if maxx<104.0421 or maxy<30.6526 or minx>104.1287 or miny>30.7271:
            drop_index.append(i)
    roads=roads.drop(index=drop_index)

    roads.to_file("./boundary_shapefile/boundary.shp", driver="ESRI Shapefile", encoding="utf-8")

def save_json(graph):
    with open("./boundary_shapefile/nodes.json", "w") as f:
        json.dump(list(graph.nodes), f)

    with open("./boundary_shapefile/edges.json", "w") as f:
        json.dump(list(graph.edges), f)

def get_length(edge):
    geom=wkt.loads(edge["Wkt"])

    roads=gp.GeoSeries(geom)
    roads.crs=CRS("epsg:4326")
    roads=roads.to_crs("epsg:2432")

    return roads.length[0]

if __name__ == "__main__":
    graph=nx.read_shp("./boundary_shapefile/boundary.shp")

    p1=(104.08143, 30.69705)
    # p2=(104.08145, 30.69757)
    p2=(104.03855, 30.71273)

    # try:
    #     length, path=nx.bidirectional_dijkstra(graph, p1, p2, weight=lambda a, b, x: get_length(x))

    #     print(length, path)
    # except nx.exception.NetworkXNoPath:
    #     print("Cannot reach")

    length_dict, path_dict=nx.single_source_dijkstra(graph, p1, weight=lambda a, b, x: get_length(x))
    print(path_dict)
    print(length_dict)

    # print(graph.nodes)
