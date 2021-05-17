import math
import sys
from typing import NoReturn

sys.path.append("../")

import Supersegment.supersegment as ss
import numpy as np
import TTI_computing.TTI_calculation as tti
import flask
import json
import networkx as nx
import shapely.wkt as wkt
import geopandas as gp

from pyproj import CRS
from urllib.parse import unquote
import time
from shapely.geometry import MultiPoint, Point


MAX_INT=999

app=flask.Flask(__name__)

graph=nx.read_shp("./boundary_shapefile/boundary.shp")

# length_dict=None
# path_dict=None

# source=None
# target=None

roads=None

buffer_distance = 0.00004
num_of_cars = 1000
TTI_interval = 120

@app.route("/")
def index():
    return flask.render_template("chengdu_map.html")

@app.route("/show_supersegment", methods=["GET"])
def show_supersegment():
    with open("../supersegment_output/supersegment_result_all.json", "r") as f:
        response=json.load(f)

    return json.dumps(response)

def get_length(edge):
    geom=wkt.loads(edge["Wkt"])

    roads=gp.GeoSeries(geom)
    roads.crs=CRS("epsg:4326")
    roads=roads.to_crs("epsg:2432")

    return roads.length[0]

def get_manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def get_matched_points(p1, p2):
    source=(MAX_INT, MAX_INT)
    target=(MAX_INT, MAX_INT)
    for point in graph:
        if get_manhattan_distance(point, p1)<get_manhattan_distance(source, p1):
            source=point
        if get_manhattan_distance(point, p2)<get_manhattan_distance(target, p2):
            target=point

    return source, target

# @app.route("/get_points", methods=["GET"])
# def get_points():
#     with open("./boundary_shapefile/nodes.json", "r") as f:
#         response=json.load(f)

#     return json.dumps(response)

# @app.route("/get_reachable_points", methods=["GET"])
# def dijkstra_source_point():
#     global source, length_dict, path_dict

#     p1=flask.request.args.get("p1")
#     p1=unquote(p1)
#     p1=p1.split(",")
#     p1[0]=float(p1[0])
#     p1[1]=float(p1[1])
#     p1=tuple(p1)

#     source=p1
    
#     length_dict, path_dict=nx.single_source_dijkstra(graph, source, weight=lambda a, b, x: get_length(x))

#     return json.dumps(list(length_dict.keys())[1:])

# @app.route("/get_distance", methods=["GET"])
# def get_distance():
#     global target, length_dict, path_dict

#     p2=flask.request.args.get("p2")
#     p2=unquote(p2)
#     p2=p2.split(",")
#     p2[0]=float(p2[0])
#     p2[1]=float(p2[1])
#     p2=tuple(p2)

#     target=p2

#     return json.dumps(length_dict[target])#, path_dict[target]

# def get_roads():
#     global target
    
#     path=path_dict[target]

#     roads=[]

#     for i in range(len(path)-1):
#         roads.append(wkt.loads(graph.get_edge_data(path[i], path[i+1])["Wkt"]))

#     roads=gp.GeoSeries(roads)
#     # app.logger.debug(roads)

#     return roads

@app.route("/show_roads", methods=["GET"])
def show_roads():
    global roads
    return roads.to_json()

def calculate_TTI():
    global roads
    df = ss.read_boundary()
    # road_start_index = 21
    # road_end_index = None
    # roads = df.loc[road_start_index:road_end_index, "geom"]
    #
    # # (104.0421, 30.6526) (104.1287, 30.7271)
    # drop_index = []
    # for i in roads.index:
    #     minx, miny, maxx, maxy = roads[i].bounds
    #     if maxx < 104.0421 or maxy < 30.6526 or minx > 104.1287 or miny > 30.7271:
    #         drop_index.append(i)
    # roads = roads.drop(index=drop_index)
    #
    # tmp_road=get_roads()
    # to_calcul_road = []
    # for each_tmp in tmp_road:
    #     minx, miny, maxx, maxy = each_tmp.bounds
    #     p = Point(minx,miny)
    #     for j in range(len(roads)):
    #         road = roads.iloc[j]
    #         if road.contains(p):
    #             to_calcul_road.append(road)
    #             break

    # 首先要计算这个路段到底对应的是哪个真实路段（更长的）
    # roads = get_roads()
    # buffer_distance = 0.00004
    # num_of_cars = 20000
    # TTI_interval = 120
    return tti.cal_TTI(roads, buffer_distance, num_of_cars, timer=False, plot=False, TTI_interval=TTI_interval)

def calculate_TTE(distance):
    TTI, free_speed_list, _ = calculate_TTI()
    query_time = time.localtime(time.time())
    query_minutes = query_time.tm_hour * 60 + query_time.tm_min
    
    tte_list = []
    for i,each_TTI in enumerate(TTI):
        speed = free_speed_list[i] / each_TTI[1][int(query_minutes/TTI_interval)]
        if math.isnan(speed):
            speed = tti.get_TTIspeed()
        tte = distance/speed
        tte_list.append(tte)
    app.logger.debug(tte_list)
    return sum(tte_list)

@app.route("/TTE_result", methods=["GET"])
def get_result():
    p1=(float(flask.request.args.get("lng1")), float(flask.request.args.get("lat1")))
    p2=(float(flask.request.args.get("lng2")), float(flask.request.args.get("lat2")))

    source, target=get_matched_points(p1, p2)

    distance, path=nx.single_source_dijkstra(graph, source, target, weight=lambda a, b, x: get_length(x))

    app.logger.debug(distance)
    app.logger.debug(path)

    global roads
    roads=[]
    for i in range(len(path)-1):
        roads.append(wkt.loads(graph.get_edge_data(path[i], path[i+1])["Wkt"]))
    roads=gp.GeoSeries(roads)

    tte=calculate_TTE(distance)

    return json.dumps([tte, distance])

if __name__ == "__main__":
    app.run(debug=True)

# osmnx