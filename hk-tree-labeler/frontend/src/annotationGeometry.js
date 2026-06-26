export function clamp(value, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

export function boxToPixels(box, width, height) {
  return {
    x: (box.x_center - box.width / 2) * width,
    y: (box.y_center - box.height / 2) * height,
    w: box.width * width,
    h: box.height * height,
  };
}

export function boxToPolygon(box) {
  const x1 = clamp(box.x_center - box.width / 2);
  const y1 = clamp(box.y_center - box.height / 2);
  const x2 = clamp(box.x_center + box.width / 2);
  const y2 = clamp(box.y_center + box.height / 2);
  return {
    class_id: box.class_id || 0,
    points: [
      [x1, y1],
      [x2, y1],
      [x2, y2],
      [x1, y2],
    ],
  };
}

export function polygonToPixels(polygon, width, height) {
  return (polygon || []).map(([x, y]) => [x * width, y * height]);
}

export function pixelsToPoint(point, width, height) {
  return [clamp(point.x / width), clamp(point.y / height)];
}

export function candidateToPolygon(candidate) {
  if (candidate?.polygon?.length >= 3) {
    return {
      class_id: 0,
      points: candidate.polygon.map(([x, y]) => [clamp(Number(x)), clamp(Number(y))]),
    };
  }
  if (candidate?.box) return boxToPolygon(candidate.box);
  return null;
}

export function candidatePolygons(candidates, limit = 1) {
  return candidates
    .map((candidate) => candidateToPolygon(candidate))
    .filter(Boolean)
    .slice(0, limit);
}

export function polygonBounds(points, width, height) {
  const pixels = polygonToPixels(points, width, height);
  if (!pixels.length) return null;
  const xs = pixels.map(([x]) => x);
  const ys = pixels.map(([, y]) => y);
  return {
    x: Math.min(...xs),
    y: Math.min(...ys),
    w: Math.max(...xs) - Math.min(...xs),
    h: Math.max(...ys) - Math.min(...ys),
  };
}

export function pointInPolygon(polygon, x, y) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    const intersects = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / ((yj - yi) || 1e-6) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}

export function candidateHitTest(candidates, x, y, width, height) {
  for (let i = candidates.length - 1; i >= 0; i -= 1) {
    const candidate = candidates[i];
    const polygon = polygonToPixels(candidate.polygon, width, height);
    if (polygon.length >= 3 && pointInPolygon(polygon, x, y)) return i;
    if (candidate.box) {
      const rect = boxToPixels(candidate.box, width, height);
      if (x >= rect.x && x <= rect.x + rect.w && y >= rect.y && y <= rect.y + rect.h) return i;
    }
  }
  return -1;
}

export function vertexHitTest(polygons, x, y, width, height) {
  for (let polygonIndex = polygons.length - 1; polygonIndex >= 0; polygonIndex -= 1) {
    const points = polygonToPixels(polygons[polygonIndex].points, width, height);
    for (let vertexIndex = points.length - 1; vertexIndex >= 0; vertexIndex -= 1) {
      const [vx, vy] = points[vertexIndex];
      if (Math.hypot(x - vx, y - vy) <= 11) return { polygonIndex, vertexIndex };
    }
  }
  return null;
}

export function polygonHitTest(polygons, x, y, width, height) {
  for (let i = polygons.length - 1; i >= 0; i -= 1) {
    const points = polygonToPixels(polygons[i].points, width, height);
    if (points.length >= 3 && pointInPolygon(points, x, y)) return i;
  }
  return -1;
}

export function contextMenuHitTest({ polygons, candidates, x, y, width, height }) {
  const polygonIndex = polygonHitTest(polygons, x, y, width, height);
  if (polygonIndex >= 0) return { type: "polygon", index: polygonIndex };

  const candidateIndex = candidateHitTest(candidates, x, y, width, height);
  if (candidateIndex >= 0) return { type: "candidate", index: candidateIndex };

  return null;
}

export function buildFreehandPolygon(points, width, height, minPixelDistance = 6) {
  const normalized = [];
  for (const point of points) {
    const current = Array.isArray(point) ? point : pixelsToPoint(point, width, height);
    const previous = normalized[normalized.length - 1];
    if (!previous) {
      normalized.push(current);
      continue;
    }
    const dx = (current[0] - previous[0]) * width;
    const dy = (current[1] - previous[1]) * height;
    if (Math.hypot(dx, dy) >= minPixelDistance) normalized.push(current);
  }
  if (normalized.length < 3) return null;
  return { class_id: 0, points: normalized };
}
