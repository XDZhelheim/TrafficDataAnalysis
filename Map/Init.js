var chengdu_map = L.map(
    "chengdu_map",
    {
        center: [30.67, 104.07],
        crs: L.CRS.EPSG3857,
        zoom: 10,
        zoomControl: true,
        preferCanvas: false,
    }
);





var tile_layer_c8d3c89666a7429984d93db24d2dafb6 = L.tileLayer(
    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    {"attribution": "Data by \u0026copy; \u003ca href=\"http://openstreetmap.org\"\u003eOpenStreetMap\u003c/a\u003e, under \u003ca href=\"http://www.openstreetmap.org/copyright\"\u003eODbL\u003c/a\u003e.", "detectRetina": false, "maxNativeZoom": 18, "maxZoom": 18, "minZoom": 0, "noWrap": false, "opacity": 1, "subdomains": "abc", "tms": false}
).addTo(chengdu_map);


function newMarker(e){
    var new_mark = L.marker().setLatLng(e.latlng).addTo(chengdu_map);
    new_mark.dragging.enable();
    new_mark.on('dblclick', function(e){ chengdu_map.removeLayer(e.target)})
    var lat = e.latlng.lat.toFixed(4),
        lng = e.latlng.lng.toFixed(4);
    new_mark.bindPopup("Latitude: " + lat + "<br>Longitude: " + lng );
};
chengdu_map.on('click', newMarker);