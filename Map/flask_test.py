import sys
sys.path.append("../")

import folium
import Supersegment.supersegment as ss
from flask import Flask

app=Flask(__name__)

m=folium.Map(location=[30.6669, 104.0655],
            crs="EPSG3857",
            zoom_start=17,
            tiles="http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}", # 路网图
            # tiles="http://webst03.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}", # 影像图
            attr="default"
            )

m.add_child(folium.ClickForMarker())

@app.route("/")
def index():
    return m._repr_html_()

@app.route("/supersegment", methods=["GET"])
def show_supersegment():
    segment_json_path="D:\\Codes\\PythonWorkspace\\TrafficDataAnalysis\\supersegment_output\\supersegment_result.json"
    folium.GeoJson(data=segment_json_path, name="supersegment").add_to(m)

    folium.LayerControl().add_to(m)

    return m._repr_html_()

if __name__ == "__main__":
    app.run(debug=True)
