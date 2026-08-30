"""
Generates a self-contained HTML/JS live-orbit component: continuous
client-side SGP4 propagation via satellite.js (no server round-trip, so
motion isn't tied to any Streamlit refresh interval), rendered on an
interactive orthographic globe via d3.js (drag to rotate, scroll to zoom),
with click-to-show-orbit-path like CelesTrak.

Verified working in a real browser as of the equirectangular version;
this update switches the projection and adds interactivity on top of
that confirmed-working foundation.
"""

import json

from tle_formatter import row_to_tle


def build_satellite_json(orbital_data_df):
    satellites = []
    for _, row in orbital_data_df.iterrows():
        try:
            line1, line2 = row_to_tle(row)
        except Exception:
            continue

        satellites.append({
            "name": row["OBJECT_NAME"],
            "norad_id": int(row["NORAD_CAT_ID"]),
            "line1": line1,
            "line2": line2,
            "type": "DEBRIS" if "DEB" in str(row["OBJECT_NAME"]).upper() else "SATELLITE",
        })
    return json.dumps(satellites)


def build_live_orbit_html(orbital_data_df, width=900, height=550, update_interval_ms=500):
    satellites_json = build_satellite_json(orbital_data_df)

    html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://unpkg.com/satellite.js/dist/satellite.min.js"></script>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script src="https://unpkg.com/topojson-client@3"></script>
<style>
  body { margin: 0; background: #0b1220; font-family: sans-serif; }
  #map-container { position: relative; }
  svg { background: #0b1220; cursor: grab; }
  svg:active { cursor: grabbing; }
  .globe-sphere { fill: #0f1c30; stroke: #334155; stroke-width: 1; }
  .country { fill: #1e293b; stroke: #334155; stroke-width: 0.5; }
  .graticule { fill: none; stroke: #1e293b; stroke-width: 0.5; }
  .sat-marker { cursor: pointer; }
  .sat-debris { fill: #ef4444; }
  .sat-satellite { fill: #22d3ee; }
  .sat-selected { fill: #f59e0b; stroke: white; stroke-width: 1.5; }
  .orbit-path { fill: none; stroke: #f59e0b; stroke-width: 1.5; opacity: 0.8; }
  #info-box {
    position: absolute; top: 10px; left: 10px; color: #e2e8f0;
    background: rgba(15,23,42,0.85); padding: 8px 12px; border-radius: 6px;
    font-size: 13px; pointer-events: none; max-width: 300px;
  }
  #clock { position: absolute; top: 10px; right: 10px; color: #94a3b8; font-size: 12px; }
  #hint { position: absolute; bottom: 10px; right: 10px; color: #64748b; font-size: 11px; }
  #debug-box {
    position: absolute; bottom: 10px; left: 10px; color: #94a3b8;
    background: rgba(15,23,42,0.85); padding: 6px 10px; border-radius: 6px;
    font-size: 11px; font-family: monospace; white-space: pre-wrap; max-height: 100px;
    max-width: 60%; overflow-y: auto;
  }
</style>
</head>
<body>
<div id="map-container">
  <svg id="map" width="__WIDTH__" height="__HEIGHT__"></svg>
  <div id="info-box">Click a satellite to show its orbit path. Red = debris, cyan = satellite.</div>
  <div id="clock"></div>
  <div id="hint">Drag to rotate &middot; Scroll to zoom</div>
  <div id="debug-box">Loading...</div>
</div>

<script>
  if (typeof satellite === "undefined") {
    document.write('<script src="https://cdn.jsdelivr.net/npm/satellite.js/dist/satellite.min.js"><\\/script>');
  }
</script>

<script>
const WIDTH = __WIDTH__;
const HEIGHT = __HEIGHT__;
const UPDATE_INTERVAL_MS = __UPDATE_INTERVAL_MS__;
const satelliteData = __SATELLITES_JSON__;

const svg = d3.select("#map");
const debugBox = document.getElementById("debug-box");
function debugLog(msg) {
  console.log("[live-orbit]", msg);
  debugBox.textContent = debugBox.textContent + msg + "\\n";
}

// --- Orthographic (globe) projection, with drag-to-rotate and scroll-to-zoom ---
let baseScale = Math.min(WIDTH, HEIGHT) / 2 - 10;
let currentRotation = [0, -20, 0];  // slight tilt for a nicer default view

const projection = d3.geoOrthographic()
  .scale(baseScale)
  .translate([WIDTH / 2, HEIGHT / 2])
  .rotate(currentRotation)
  .clipAngle(90);

const path = d3.geoPath(projection);

const sphereLayer = svg.append("g").attr("id", "sphere-layer");
const graticuleLayer = svg.append("g").attr("id", "graticule-layer");
const countryLayer = svg.append("g").attr("id", "country-layer");
const orbitLayer = svg.append("g").attr("id", "orbit-layer");
const markersLayer = svg.append("g").attr("id", "markers-layer");

sphereLayer.append("path")
  .datum({ type: "Sphere" })
  .attr("class", "globe-sphere");

graticuleLayer.append("path")
  .datum(d3.geoGraticule()())
  .attr("class", "graticule");

let worldFeatures = null;

d3.json("https://unpkg.com/world-atlas@2/countries-110m.json").then(function(world) {
  worldFeatures = topojson.feature(world, world.objects.countries).features;
  countryLayer.selectAll("path")
    .data(worldFeatures)
    .join("path")
    .attr("class", "country");
  redrawStatic();
}).catch(function(err) {
  debugLog("World map failed to load: " + err.message + " (globe outline/markers will still work)");
});

function redrawStatic() {
  sphereLayer.select("path").attr("d", path);
  graticuleLayer.select("path").attr("d", path);
  if (worldFeatures) {
    countryLayer.selectAll("path").attr("d", path);
  }
}

// --- Drag to rotate (but not when the gesture starts on a satellite
// marker — otherwise the drag behavior swallows the click event before
// the marker's own click handler ever fires) ---
const drag = d3.drag()
  .filter(function(event) {
    return !event.target.classList.contains("sat-marker");
  })
  .clickDistance(4)  // tiny accidental movements still count as a click, not a drag
  .on("drag", function(event) {
    const rotate = projection.rotate();
    const k = 75 / projection.scale();
    projection.rotate([rotate[0] + event.dx * k, rotate[1] - event.dy * k, rotate[2]]);
    redrawStatic();
    updateMarkers();
    if (selectedSatEntry) drawOrbitPath(selectedSatEntry);
  });
svg.call(drag);

// --- Scroll to zoom ---
svg.on("wheel", function(event) {
  event.preventDefault();
  const newScale = Math.max(baseScale * 0.4, Math.min(baseScale * 4,
    projection.scale() * (event.deltaY < 0 ? 1.1 : 0.9)));
  projection.scale(newScale);
  redrawStatic();
  updateMarkers();
  if (selectedSatEntry) drawOrbitPath(selectedSatEntry);
});

// --- Parse TLEs ---
let satrecs = [];
let parseErrors = 0;

if (typeof satellite === "undefined") {
  debugLog("ERROR: satellite.js did not load. Check your internet connection.");
} else {
  satelliteData.forEach(function(sat) {
    try {
      const rec = satellite.twoline2satrec(sat.line1, sat.line2);
      if (rec && !isNaN(rec.no)) {
        satrecs.push({ name: sat.name, norad_id: sat.norad_id, type: sat.type, satrec: rec });
      } else {
        parseErrors++;
      }
    } catch (err) {
      parseErrors++;
      debugLog("Exception parsing TLE for " + sat.name + ": " + err.message);
    }
  });
  debugLog("Parsed " + satrecs.length + "/" + satelliteData.length + " satellites" +
           (parseErrors > 0 ? " (" + parseErrors + " failed)" : "."));
}

let selectedNoradId = null;
let selectedSatEntry = null;

function currentPositions() {
  const now = new Date();
  const gmst = satellite.gstime(now);

  return satrecs.map(function(s) {
    const posVel = satellite.propagate(s.satrec, now);
    if (!posVel || !posVel.position) return null;
    const geo = satellite.eciToGeodetic(posVel.position, gmst);
    return {
      name: s.name, norad_id: s.norad_id, type: s.type,
      lat: satellite.degreesLat(geo.latitude),
      lon: satellite.degreesLong(geo.longitude),
    };
  }).filter(function(p) { return p !== null; });
}

// A point is on the visible (front) hemisphere if its angular distance from
// the current view center is under 90 degrees.
function isVisible(lon, lat) {
  const rotate = projection.rotate();
  const center = [-rotate[0], -rotate[1]];
  return d3.geoDistance([lon, lat], center) < Math.PI / 2;
}

function updateMarkers() {
  try {
    const positions = currentPositions().filter(function(d) { return isVisible(d.lon, d.lat); });

    const markers = markersLayer.selectAll("circle")
      .data(positions, function(d) { return d.norad_id; });

    markers.join(
      function(enter) {
        return enter.append("circle")
          .attr("class", function(d) {
            return "sat-marker " + (d.norad_id === selectedNoradId ? "sat-selected"
                   : d.type === "DEBRIS" ? "sat-debris" : "sat-satellite");
          })
          .attr("r", function(d) { return d.norad_id === selectedNoradId ? 4 : 2.5; })
          .on("click", function(event, d) { onSatelliteClick(d); })
          .append("title")
          .text(function(d) { return d.name + " (NORAD " + d.norad_id + ")"; });
      },
      function(update) {
        return update
          .attr("class", function(d) {
            return "sat-marker " + (d.norad_id === selectedNoradId ? "sat-selected"
                   : d.type === "DEBRIS" ? "sat-debris" : "sat-satellite");
          })
          .attr("r", function(d) { return d.norad_id === selectedNoradId ? 4 : 2.5; });
      },
      function(exit) { exit.remove(); }
    )
    .attr("cx", function(d) { const c = projection([d.lon, d.lat]); return c ? c[0] : -100; })
    .attr("cy", function(d) { const c = projection([d.lon, d.lat]); return c ? c[1] : -100; });

    document.getElementById("clock").textContent =
      new Date().toISOString().replace("T", " ").slice(0, 19) + " UTC";
  } catch (err) {
    debugLog("ERROR in updateMarkers: " + err.message);
  }
}

function drawOrbitPath(satEntry) {
  orbitLayer.selectAll("*").remove();

  const periodMinutes = (2 * Math.PI) / satEntry.satrec.no;
  const now = new Date();
  const nSamples = 150;
  const points = [];

  for (let i = 0; i <= nSamples; i++) {
    const t = new Date(now.getTime() + (i / nSamples) * periodMinutes * 60000);
    const posVel = satellite.propagate(satEntry.satrec, t);
    if (!posVel || !posVel.position) continue;
    const gmst = satellite.gstime(t);
    const geo = satellite.eciToGeodetic(posVel.position, gmst);
    points.push([satellite.degreesLong(geo.longitude), satellite.degreesLat(geo.latitude)]);
  }

  // Build a proper GeoJSON LineString and let d3.geoPath handle hemisphere
  // clipping automatically (it correctly trims the line at the horizon for
  // an orthographic projection, rather than drawing through the globe).
  orbitLayer.append("path")
    .datum({ type: "LineString", coordinates: points })
    .attr("class", "orbit-path")
    .attr("d", path);
}

function onSatelliteClick(d) {
  if (selectedNoradId === d.norad_id) {
    selectedNoradId = null;
    selectedSatEntry = null;
    orbitLayer.selectAll("*").remove();
    document.getElementById("info-box").textContent =
      "Click a satellite to show its orbit path. Red = debris, cyan = satellite.";
    updateMarkers();
    return;
  }

  selectedNoradId = d.norad_id;
  selectedSatEntry = satrecs.find(function(s) { return s.norad_id === d.norad_id; });
  if (selectedSatEntry) drawOrbitPath(selectedSatEntry);

  document.getElementById("info-box").innerHTML =
    "<b>" + d.name + "</b><br>NORAD " + d.norad_id + " (" + d.type + ")" +
    "<br>Lat " + d.lat.toFixed(2) + "&deg;, Lon " + d.lon.toFixed(2) + "&deg;" +
    "<br><i>Click again to deselect</i>";
  updateMarkers();
}

updateMarkers();
setInterval(updateMarkers, UPDATE_INTERVAL_MS);
</script>
</body>
</html>
"""

    html = html.replace("__WIDTH__", str(width))
    html = html.replace("__HEIGHT__", str(height))
    html = html.replace("__UPDATE_INTERVAL_MS__", str(update_interval_ms))
    html = html.replace("__SATELLITES_JSON__", satellites_json)

    return html