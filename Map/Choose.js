let a = 0;
let move = false;
let pointer1 = null;
let pointer2 = null;

const chengdu_map = L.map(
    "chengdu_map",
    {
        center: [30.67, 104.07],
        crs: L.CRS.EPSG3857,
        zoom: 10,
        zoomControl: true,
        preferCanvas: false,
    }
);

L.tileLayer(
    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    {
        "attribution": "Data by \u0026copy; \u003ca href=\"http://openstreetmap.org\"\u003eOpenStreetMap\u003c/a\u003e, under \u003ca href=\"http://www.openstreetmap.org/copyright\"\u003eODbL\u003c/a\u003e.",
        "detectRetina": false,
        "maxNativeZoom": 18,
        "maxZoom": 18,
        "minZoom": 0,
        "noWrap": false,
        "opacity": 1,
        "subdomains": "abc",
        "tms": false
    }
).addTo(chengdu_map);

function newMarker(e) {
    if (a === 2) {
        return;
    }
    a++;
    const new_mark = L.marker().setLatLng(e.latlng).addTo(chengdu_map);
    new_mark.dragging.enable();
    if (pointer1 == null) {
        new_mark._index_ = 1;
        pointer1 = new_mark;
    } else {
        new_mark._index_ = 2;
        pointer2 = new_mark;
    }
    new_mark.on('dblclick', function (e) {
        chengdu_map.removeLayer(e.target)
        document.getElementById('p' + new_mark._index_).innerText = "";
        if (new_mark._index_ === 1) {
            pointer1 = null;
        } else {
            pointer2 = null;
        }
        a--;
    })
    new_mark.on('mousemove', function (e) {
        if (move)
            document.getElementById('p' + new_mark._index_).innerText = "(" + e.latlng.lat.toFixed(4) + "," + e.latlng.lng.toFixed(4) + ")";
    })
    new_mark.on('mouseup', function () {
        move = false
    })
    new_mark.on('mousedown', function () {
        move = true
    })
    const lat = e.latlng.lat.toFixed(4),
        lng = e.latlng.lng.toFixed(4);
    new_mark.bindPopup("Latitude: " + lat + "<br>Longitude: " + lng);
    document.getElementById('p' + new_mark._index_).innerText = "(" + lat + "," + lng + ")";
}
chengdu_map.on('click', newMarker);