import test from "node:test";
import assert from "node:assert/strict";

import { appendCandidatePolygon, buildFreehandPolygon, candidatePolygons, contextMenuHitTest, shouldRefreshHoverPoint } from "./annotationGeometry.js";

test("buildFreehandPolygon closes a freehand path and skips dense duplicate points", () => {
  const polygon = buildFreehandPolygon(
    [
      [0.1, 0.1],
      [0.101, 0.101],
      [0.3, 0.1],
      [0.3, 0.3],
      [0.1, 0.3],
    ],
    640,
    640,
  );

  assert.deepEqual(polygon, {
    class_id: 0,
    points: [
      [0.1, 0.1],
      [0.3, 0.1],
      [0.3, 0.3],
      [0.1, 0.3],
    ],
  });
});

test("contextMenuHitTest prefers editable polygons and can target candidate masks", () => {
  const polygons = [{ points: [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]] }];
  const candidates = [{ polygon: [[0.6, 0.6], [0.9, 0.6], [0.9, 0.9], [0.6, 0.9]] }];

  assert.deepEqual(contextMenuHitTest({ polygons, candidates, x: 128, y: 128, width: 640, height: 640 }), {
    type: "polygon",
    index: 0,
  });
  assert.deepEqual(contextMenuHitTest({ polygons, candidates, x: 480, y: 480, width: 640, height: 640 }), {
    type: "candidate",
    index: 0,
  });
  assert.equal(contextMenuHitTest({ polygons, candidates, x: 12, y: 600, width: 640, height: 640 }), null);
});

test("candidatePolygons converts masks into ordinary editable label polygons", () => {
  const polygons = candidatePolygons(
    [
      { polygon: [[0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.2, 0.4]] },
      { polygon: [[0.6, 0.6], [0.8, 0.6], [0.8, 0.8], [0.6, 0.8]] },
    ],
    1,
  );

  assert.deepEqual(polygons, [
    {
      class_id: 0,
      points: [[0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.2, 0.4]],
    },
  ]);
});

test("appendCandidatePolygon keeps existing labels and appends SAM candidates", () => {
  const existing = [{ class_id: 0, points: [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]] }];
  const next = appendCandidatePolygon(existing, {
    polygon: [[0.5, 0.5], [0.7, 0.5], [0.7, 0.7], [0.5, 0.7]],
  });

  assert.deepEqual(next, [
    { class_id: 0, points: [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]] },
    { class_id: 0, points: [[0.5, 0.5], [0.7, 0.5], [0.7, 0.7], [0.5, 0.7]] },
  ]);
});

test("shouldRefreshHoverPoint throttles tiny hover movement", () => {
  assert.equal(shouldRefreshHoverPoint(null, [0.5, 0.5]), true);
  assert.equal(shouldRefreshHoverPoint([0.5, 0.5], [0.505, 0.505]), false);
  assert.equal(shouldRefreshHoverPoint([0.5, 0.5], [0.53, 0.5]), true);
});
