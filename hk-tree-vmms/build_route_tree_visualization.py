from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay-geojson", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overlay = json.loads(args.overlay_geojson.read_text(encoding="utf-8"))
    routes = []
    trees = []
    for feature in overlay["features"]:
        props = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        if props.get("feature_kind") == "route" and geometry.get("type") == "LineString":
            routes.append(
                {
                    "id": props.get("stream_id"),
                    "coordinates": [
                        [round(float(point[0]), 7), round(float(point[1]), 7)]
                        for point in geometry["coordinates"]
                    ],
                }
            )
        elif props.get("feature_kind") == "tree" and geometry.get("type") == "Point":
            trees.append(
                {
                    "x": round(float(geometry["coordinates"][0]), 7),
                    "y": round(float(geometry["coordinates"][1]), 7),
                    "d": round(float(props["distance_to_route_m"]), 1),
                    "source": props.get("source_dataset"),
                    "id": props.get("source_tree_id"),
                    "species": props.get("species_name"),
                    "route": props.get("nearest_route_id"),
                }
            )
    payload = json.dumps(
        {"routes": routes, "trees": trees},
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")

    fragment = f'''<div id="vmms-tree-route-viz">
  <h2>轨迹附近树点分布</h2>
  <div class="viz-controls" aria-label="树点筛选">
    <label class="form-label" for="vmms-route-filter">路线
      <select class="form-select" id="vmms-route-filter"><option value="all">全部路线</option></select>
    </label>
    <label class="form-label" for="vmms-distance-filter">最大距离
      <select class="form-select" id="vmms-distance-filter">
        <option value="10">10 m</option><option value="20">20 m</option>
        <option value="30">30 m</option><option value="50" selected>50 m</option>
        <option value="100">100 m</option>
      </select>
    </label>
    <div class="vmms-source-controls" role="group" aria-label="树点来源">
      <label class="form-check"><input class="form-check-input" type="checkbox" value="csdi_roadside" checked> <span class="form-check-label">CSDI 路旁树</span></label>
      <label class="form-check"><input class="form-check-input" type="checkbox" value="major_parks" checked> <span class="form-check-label">主要公园</span></label>
      <label class="form-check"><input class="form-check-input" type="checkbox" value="tree20250604" checked> <span class="form-check-label">2025 树表</span></label>
    </div>
  </div>
  <div class="vmms-map-status text-small" aria-live="polite"></div>
  <div class="vmms-map-wrap">
    <svg role="img" aria-label="VMMS采集路线与附近香港树点空间分布图"></svg>
    <div class="tooltip" role="tooltip" hidden></div>
  </div>
  <div class="vmms-map-legend text-small" aria-label="图例">
    <span><i class="vmms-line-key"></i>采集路线</span>
    <span><i class="vmms-dot-key vmms-csdi"></i>CSDI 路旁树</span>
    <span><i class="vmms-dot-key vmms-parks"></i>主要公园</span>
    <span><i class="vmms-dot-key vmms-2025"></i>2025 树表</span>
  </div>
</div>
<style>
#vmms-tree-route-viz {{ width:100%; color:var(--foreground); }}
#vmms-tree-route-viz h2 {{ margin:0 0 10px; font-weight:500; }}
#vmms-tree-route-viz .viz-controls {{ align-items:end; }}
#vmms-tree-route-viz .form-label {{ min-width:150px; }}
#vmms-tree-route-viz .vmms-source-controls {{ display:flex; flex-wrap:wrap; gap:6px 14px; align-items:center; padding-bottom:7px; }}
#vmms-tree-route-viz .vmms-map-status {{ margin:8px 0; color:var(--muted-foreground); }}
#vmms-tree-route-viz .vmms-map-wrap {{ position:relative; width:100%; min-height:390px; }}
#vmms-tree-route-viz svg {{ display:block; width:100%; min-height:390px; }}
#vmms-tree-route-viz .vmms-frame {{ fill:transparent; stroke:var(--border); stroke-width:1; }}
#vmms-tree-route-viz .vmms-grid {{ fill:none; stroke:var(--border); stroke-width:1; opacity:.45; }}
#vmms-tree-route-viz .vmms-route {{ fill:none; stroke:var(--viz-series-1); stroke-width:2.5; stroke-linecap:round; stroke-linejoin:round; }}
#vmms-tree-route-viz .vmms-route-label {{ fill:var(--foreground); font-size:12px; font-weight:500; paint-order:stroke; stroke:var(--background); stroke-width:4px; stroke-linejoin:round; }}
#vmms-tree-route-viz .vmms-tree {{ stroke:var(--background); stroke-width:.8; }}
#vmms-tree-route-viz .vmms-tree[data-source="csdi_roadside"] {{ fill:var(--viz-series-2); }}
#vmms-tree-route-viz .vmms-tree[data-source="major_parks"] {{ fill:var(--viz-series-3); }}
#vmms-tree-route-viz .vmms-tree[data-source="tree20250604"] {{ fill:var(--viz-series-4); }}
#vmms-tree-route-viz .tooltip {{ position:absolute; pointer-events:none; max-width:280px; padding:8px 10px; background:var(--popover); color:var(--popover-foreground); border:1px solid var(--border); border-radius:6px; box-shadow:0 4px 16px color-mix(in srgb,var(--foreground) 16%,transparent); }}
#vmms-tree-route-viz .vmms-map-legend {{ display:flex; flex-wrap:wrap; gap:8px 16px; margin-top:8px; color:var(--muted-foreground); }}
#vmms-tree-route-viz .vmms-map-legend span {{ display:inline-flex; align-items:center; gap:6px; }}
#vmms-tree-route-viz .vmms-line-key {{ width:22px; height:3px; background:var(--viz-series-1); }}
#vmms-tree-route-viz .vmms-dot-key {{ width:9px; height:9px; border-radius:50%; }}
#vmms-tree-route-viz .vmms-csdi {{ background:var(--viz-series-2); }}
#vmms-tree-route-viz .vmms-parks {{ background:var(--viz-series-3); }}
#vmms-tree-route-viz .vmms-2025 {{ background:var(--viz-series-4); }}
@media (max-width:520px) {{
  #vmms-tree-route-viz .form-label {{ min-width:120px; flex:1; }}
  #vmms-tree-route-viz .vmms-map-wrap, #vmms-tree-route-viz svg {{ min-height:330px; }}
}}
</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script type="application/json" id="vmms-tree-route-data">{payload}</script>
<script>
(() => {{
  const root = document.getElementById('vmms-tree-route-viz');
  const data = JSON.parse(document.getElementById('vmms-tree-route-data').textContent);
  const svg = d3.select(root.querySelector('svg'));
  const wrap = root.querySelector('.vmms-map-wrap');
  const status = root.querySelector('.vmms-map-status');
  const tooltip = root.querySelector('.tooltip');
  const routeSelect = root.querySelector('#vmms-route-filter');
  const distanceSelect = root.querySelector('#vmms-distance-filter');
  const routeLabels = {{
    hewentian_pano:'何文田', jianshazui_pano_1:'尖沙咀 1',
    jianshazui_pano_2:'尖沙咀 2', stubbs_pano_0:'Stubbs 0', stubbs_pano_1:'Stubbs 1'
  }};
  data.routes.forEach(route => {{
    const option = document.createElement('option'); option.value=route.id;
    option.textContent=routeLabels[route.id] || route.id; routeSelect.appendChild(option);
  }});
  const featureForRoute = route => ({{type:'Feature',geometry:{{type:'LineString',coordinates:route.coordinates}}}});
  function selectedSources() {{ return new Set([...root.querySelectorAll('.vmms-source-controls input:checked')].map(x=>x.value)); }}
  function showTooltip(event, tree) {{
    const species = tree.species || '未记录树种';
    tooltip.innerHTML = `<strong>${{species}}</strong><br>${{tree.id}} · ${{tree.d.toFixed(1)}} m<br>${{routeLabels[tree.route] || tree.route}} · ${{tree.source}}`;
    tooltip.hidden=false;
    const box=wrap.getBoundingClientRect(); let left=event.clientX-box.left+12; let top=event.clientY-box.top+12;
    if(left+tooltip.offsetWidth>box.width) left=event.clientX-box.left-tooltip.offsetWidth-12;
    if(top+tooltip.offsetHeight>box.height) top=event.clientY-box.top-tooltip.offsetHeight-12;
    tooltip.style.left=`${{Math.max(0,left)}}px`; tooltip.style.top=`${{Math.max(0,top)}}px`;
  }}
  function render() {{
    tooltip.hidden=true;
    const width=Math.max(320,Math.round(wrap.getBoundingClientRect().width));
    const height=width<520?330:Math.min(560,Math.max(390,Math.round(width*.58)));
    svg.attr('viewBox',`0 0 ${{width}} ${{height}}`).attr('height',height); svg.selectAll('*').remove();
    const routeId=routeSelect.value, maxDistance=+distanceSelect.value, sources=selectedSources();
    const routes=routeId==='all'?data.routes:data.routes.filter(r=>r.id===routeId);
    const trees=data.trees.filter(t=>t.d<=maxDistance && sources.has(t.source) && (routeId==='all'||t.route===routeId));
    const collection={{type:'FeatureCollection',features:routes.map(featureForRoute)}};
    const projection=d3.geoMercator().fitExtent([[38,24],[width-22,height-32]],collection);
    const path=d3.geoPath(projection);
    svg.append('rect').attr('class','vmms-frame').attr('x',.5).attr('y',.5).attr('width',width-1).attr('height',height-1);
    const bounds=d3.geoBounds(collection), lonStep=.002, latStep=.002;
    const graticule=d3.geoGraticule().extent(bounds).step([lonStep,latStep]);
    svg.append('path').datum(graticule()).attr('class','vmms-grid').attr('d',path);
    svg.append('g').selectAll('path').data(routes).join('path').attr('class','vmms-route').attr('d',d=>path(featureForRoute(d)));
    const projectedTrees=trees.map(tree=>({{tree,point:projection([tree.x,tree.y])}})).filter(d=>d.point);
    svg.append('g').selectAll('circle').data(projectedTrees).join('circle')
      .attr('class','vmms-tree').attr('data-source',d=>d.tree.source)
      .attr('cx',d=>d.point[0]).attr('cy',d=>d.point[1]).attr('r',d=>d.tree.d<=20?3.5:2.7);
    const labels=svg.append('g').selectAll('text').data(routes).join('text').attr('class','vmms-route-label')
      .attr('x',d=>projection(d.coordinates[Math.floor(d.coordinates.length/2)])[0]+5)
      .attr('y',d=>projection(d.coordinates[Math.floor(d.coordinates.length/2)])[1]-5)
      .text(d=>routeLabels[d.id]||d.id);
    if(projectedTrees.length) {{
      const delaunay=d3.Delaunay.from(projectedTrees,d=>d.point[0],d=>d.point[1]);
      svg.append('rect').attr('x',0).attr('y',0).attr('width',width).attr('height',height)
        .attr('fill','transparent').style('pointer-events','all')
        .on('pointermove',event=>{{const [x,y]=d3.pointer(event);const i=delaunay.find(x,y);const hit=projectedTrees[i];if(Math.hypot(hit.point[0]-x,hit.point[1]-y)<=22)showTooltip(event,hit.tree);else tooltip.hidden=true;}})
        .on('pointerleave',()=>tooltip.hidden=true);
    }}
    const closeCount=trees.filter(t=>t.d<=20).length;
    status.textContent=`显示 ${{trees.length}} 棵树；其中 ${{closeCount}} 棵距轨迹不超过 20 m。`;
  }}
  root.querySelectorAll('select,input').forEach(control=>control.addEventListener('change',render));
  new ResizeObserver(render).observe(wrap); render();
}})();
</script>
'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment, encoding="utf-8")
    print(f"Routes: {len(routes)}; trees: {len(trees)}; bytes: {args.output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

