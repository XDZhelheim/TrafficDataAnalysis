import sys
sys.path.append("../")

import Supersegment.supersegment as ss
import TTI_computing.TTI_calculation as tti
import flask
import json
import networkx as nx
import shapely.wkt as wkt
import geopandas as gp

from pyproj import CRS
from urllib.parse import unquote
import time

app=flask.Flask(__name__)

graph=nx.read_shp("./boundary_shapefile/boundary.shp")

length_dict=None
path_dict=None

source=None
target=None

buffer_distance = 0.00004
num_of_cars = 1000
TTI_interval = 30

@app.route("/")
def index():
    return flask.render_template("chengdu_map.html")

@app.route("/show_supersegment", methods=["GET"])
def show_supersegment():
    # ss.supersegment()

    with open("../supersegment_output/supersegment_result_all.json", "r") as f:
        response=json.load(f)

    return response

# @app.route("/road_match", methods=["GET"])
# def road_match():
#     p1=[float(flask.request.args.get("lat1")), float(flask.request.args.get("lng1"))]
#     p2=[float(flask.request.args.get("lat2")), float(flask.request.args.get("lng2"))]
#     app.logger.debug(p1)
#     app.logger.debug(p2)

#     # TODO: road match

#     with open("../supersegment_output/supersegment_result_all.json", "r") as f:
#         response=json.load(f)

#     return response

def get_length(edge):
    geom=wkt.loads(edge["Wkt"])

    roads=gp.GeoSeries(geom)
    roads.crs=CRS("epsg:4326")
    roads=roads.to_crs("epsg:2432")

    return roads.length[0]

@app.route("/get_points", methods=["GET"])
def get_points():
    with open("./boundary_shapefile/nodes.json", "r") as f:
        response=json.load(f)

    return json.dumps(response)

@app.route("/get_reachable_points", methods=["GET"])
def dijkstra_source_point():
    global source, length_dict, path_dict

    p1=flask.request.args.get("p1")
    p1=unquote(p1)
    p1=p1.split(",")
    p1[0]=float(p1[0])
    p1[1]=float(p1[1])
    p1=tuple(p1)

    source=p1
    
    length_dict, path_dict=nx.single_source_dijkstra(graph, source, weight=lambda a, b, x: get_length(x))

    return json.dumps(list(length_dict.keys())[1:])

@app.route("/get_distance", methods=["GET"])
def get_distance():
    global target, length_dict, path_dict

    p2=flask.request.args.get("p2")
    p2=unquote(p2)
    p2=p2.split(",")
    p2[0]=float(p2[0])
    p2[1]=float(p2[1])
    p2=tuple(p2)

    target=p2
    get_roads()

    return json.dumps(length_dict[target])#, path_dict[target]

def get_roads():
    global target
    
    path=path_dict[target]

    roads=[]

    for i in range(len(path)-1):
        roads.append(wkt.loads(graph.get_edge_data(path[i], path[i+1])["Wkt"]))

    roads=gp.GeoSeries(roads)
    # app.logger.debug(roads)

    return roads

@app.route("/show_roads", methods=["GET"])
def show_roads():
    roads=get_roads()

    return roads.to_json()

def calculate_TTI():
    roads=get_roads()
    buffer_distance = 0.00004
    num_of_cars = 20000
    TTI_interval = 30
    return tti.cal_TTI(roads, buffer_distance, num_of_cars, timer=False, plot=False, TTI_interval=TTI_interval)

@app.route("/TTE", methods=["GET"])
def calculate_TTE():
    global length_dict

    distance = length_dict[target]
    TTI, free_speed = calculate_TTI()
    query_time = time.localtime(time.time())
    query_minutes = query_time.tm_hour * 60 + query_time.tm_min
    tte_list = []
    for each_TTI in TTI:
        speed = free_speed / each_TTI[1][int(query_minutes/TTI_interval)]
        tte = distance/speed
        tte_list.append(tte)

    app.logger.debug(tte_list)

    return json.dumps(str(sum(tte_list)))

if __name__ == "__main__":
    app.run(debug=True)
