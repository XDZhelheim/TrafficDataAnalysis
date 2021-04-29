import sys
sys.path.append("../")

import Supersegment.supersegment as ss
import TTI_computing.TTI_calculation as tti
import flask
import json

app=flask.Flask(__name__)

@app.route("/")
def index():
    return flask.render_template("chengdu_map.html")

@app.route("/show_supersegment", methods=["GET"])
def show_supersegment():
    # ss.supersegment()

    with open("../supersegment_output/supersegment_result.json", "r") as f:
        response=json.load(f)

    return response

@app.route("/road_match", methods=["GET"])
def road_match():
    p1=[float(flask.request.args.get("lat1")), float(flask.request.args.get("lng1"))]
    p2=[float(flask.request.args.get("lat2")), float(flask.request.args.get("lng2"))]
    app.logger.debug(p1)
    app.logger.debug(p2)

    # TODO: road match

    with open("../supersegment_output/supersegment_result.json", "r") as f:
        response=json.load(f)

    return response

def calculate_distance(roads):
    return roads.distance.sum()

def calculate_TTI(roads):
    # return tti.calculate_TTI(roads, xxxxxx)
    pass

@app.route("/TTE", methods=["GET"])
def calculate_TTE(roads):
    # distance=calculate_distance(roads)
    # TTI=calculate_TTI(roads)
    # speed=free_speed*TTI*xxxxxx
    # tte=distance/speed
    # return tte
    pass

if __name__ == "__main__":
    app.run(debug=True)
