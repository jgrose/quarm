// ═══ NORT CAMERA SYSTEM ═══
// Pan/zoom with mouse wheel and drag

var camera = { x: 0, y: 0, zoom: 1, dragging: false, lastMx: 0, lastMy: 0 };

function initCamera(canvas) {
  canvas.addEventListener('wheel', function(e) {
    e.preventDefault();
    var factor = e.deltaY > 0 ? 0.92 : 1.08;
    camera.zoom = Math.max(0.2, Math.min(4, camera.zoom * factor));
  }, { passive: false });

  canvas.addEventListener('mousedown', function(e) {
    if (e.button === 0 && !getNodeAt(e)) {
      var rect = canvas.getBoundingClientRect();
      var sx = e.clientX - rect.left;
      var sy = e.clientY - rect.top;
      if (typeof isOverMinimap === 'function' && isOverMinimap(sx, sy, canvas.height)) {
        minimapMouseDown(e, canvas);
        return;
      }
      camera.dragging = true;
      camera.lastMx = e.clientX;
      camera.lastMy = e.clientY;
      canvas.style.cursor = 'grabbing';
    }
  });

  canvas.addEventListener('mousemove', function(e) {
    if (typeof _minimapDragging !== 'undefined' && _minimapDragging) {
      minimapMouseMove(e, canvas);
      return;
    }
    if (camera.dragging) {
      camera.x += (e.clientX - camera.lastMx) / camera.zoom;
      camera.y += (e.clientY - camera.lastMy) / camera.zoom;
      camera.lastMx = e.clientX;
      camera.lastMy = e.clientY;
    }
  });

  var endDrag = function() {
    camera.dragging = false;
    if (typeof minimapMouseUp === 'function') minimapMouseUp();
    canvas.style.cursor = 'grab';
  };
  canvas.addEventListener('mouseup', endDrag);
  canvas.addEventListener('mouseleave', endDrag);
}

function applyCamera(ctx, canvas) {
  ctx.translate(canvas.width / 2, canvas.height / 2);
  ctx.scale(camera.zoom * dpr, camera.zoom * dpr);
  ctx.translate(-canvas.width / (2 * dpr) + camera.x, -canvas.height / (2 * dpr) + camera.y);
}

function getVisibleRect(W, H) {
  var z = camera.zoom;
  return {
    x: W / 2 - camera.x - W / (2 * z),
    y: H / 2 - camera.y - H / (2 * z),
    w: W / z,
    h: H / z,
  };
}

function screenToWorld(sx, sy, canvas) {
  return {
    x: (sx - canvas.width / (2 * dpr)) / camera.zoom + canvas.width / (2 * dpr) - camera.x,
    y: (sy - canvas.height / (2 * dpr)) / camera.zoom + canvas.height / (2 * dpr) - camera.y,
  };
}

function getNodeAt(e) {
  var cvs = document.getElementById('canvas');
  var rect = cvs.getBoundingClientRect();
  var sx = e.clientX - rect.left;
  var sy = e.clientY - rect.top;
  var world = screenToWorld(sx, sy, cvs);
  for (var entry of nodes) {
    var node = entry[1];
    var dx = world.x - node.x;
    var dy = world.y - node.y;
    if (dx * dx + dy * dy < (node.radius + 5) * (node.radius + 5)) return node;
  }
  return null;
}
