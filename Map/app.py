import sys
sys.path.append("../")

import Supersegment.supersegment as ss
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

if __name__ == "__main__":
    app.run(debug=True)
