// ═══ TASK DEPENDENCY DAG VISUALIZATION ═══
// Renders an SVG DAG of task dependencies with topological layering.
// Receives task data from WebSocket status updates via updateDagPanel().

(function() {
  var _dagVisible = false;

  // ── Status-to-color mapping (matches C palette in colors.js) ──────────
  var _dagColors = {
    pending:                '#66ccff',
    in_progress:            '#66ccff',
    in_manager_review:      '#ffbb44',
    in_specialist_review:   '#cc88ff',
    revision:               '#ff8800',
    done:                   '#66ffaa',
    failed:                 '#ff5566'
  };

  function _dagColor(status) {
    return _dagColors[status] || '#66ccff';
  }

  // ── Topological layering ──────────────────────────────────────────────
  // Assign each task a depth (layer) based on its dependency chain.
  // Tasks with no dependencies sit at layer 0; dependents sit at
  // max(parent depth) + 1.

  function _computeLayers(tasks) {
    var taskMap = {};
    for (var i = 0; i < tasks.length; i++) {
      taskMap[tasks[i].id] = tasks[i];
    }

    var depths = {};
    var visiting = {};

    function depthOf(id) {
      if (depths[id] !== undefined) return depths[id];
      if (visiting[id]) return 0; // cycle guard
      visiting[id] = true;

      var task = taskMap[id];
      if (!task) { depths[id] = 0; return 0; }

      var deps = task.depends_on || [];
      if (deps.length === 0) { depths[id] = 0; return 0; }

      var maxParent = 0;
      for (var j = 0; j < deps.length; j++) {
        var pd = depthOf(deps[j]);
        if (pd + 1 > maxParent) maxParent = pd + 1;
      }
      depths[id] = maxParent;
      return maxParent;
    }

    for (var k = 0; k < tasks.length; k++) {
      depthOf(tasks[k].id);
    }

    // Group tasks by layer
    var layers = {};
    var maxLayer = 0;
    for (var m = 0; m < tasks.length; m++) {
      var d = depths[tasks[m].id] || 0;
      if (!layers[d]) layers[d] = [];
      layers[d].push(tasks[m]);
      if (d > maxLayer) maxLayer = d;
    }

    return { layers: layers, maxLayer: maxLayer, depths: depths };
  }

  // ── SVG helpers ───────────────────────────────────────────────────────

  function _svgEscape(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // ── Main render ───────────────────────────────────────────────────────

  function _renderDag() {
    var svg = document.getElementById('dagSvg');
    if (!svg) return;

    // Pull tasks from the active session
    var tasks = [];
    if (typeof _activeSessionId !== 'undefined' && _activeSessionId &&
        typeof _sessions !== 'undefined' && _sessions[_activeSessionId]) {
      var data = _sessions[_activeSessionId].data;
      if (data && data.tasks) tasks = data.tasks;
    }

    if (tasks.length === 0) {
      svg.innerHTML = '<text x="50%" y="50%" text-anchor="middle"' +
        ' fill="rgba(102,204,255,0.56)" font-size="11"' +
        ' font-family="Courier New, monospace">No tasks in active session</text>';
      svg.setAttribute('height', '200');
      return;
    }

    var result = _computeLayers(tasks);
    var layers = result.layers;
    var maxLayer = result.maxLayer;

    // Layout constants
    var nodeW = 120;
    var nodeH = 36;
    var layerGapX = 160;
    var nodeGapY = 52;
    var padX = 40;
    var padY = 30;

    // Compute per-task positions: layer -> X, index within layer -> Y
    var positions = {};
    var maxNodesInLayer = 0;
    for (var l = 0; l <= maxLayer; l++) {
      var layerTasks = layers[l] || [];
      if (layerTasks.length > maxNodesInLayer) maxNodesInLayer = layerTasks.length;
    }

    for (var layer = 0; layer <= maxLayer; layer++) {
      var lt = layers[layer] || [];
      var cx = padX + layer * layerGapX + nodeW / 2;
      for (var n = 0; n < lt.length; n++) {
        var cy = padY + n * nodeGapY + nodeH / 2;
        positions[lt[n].id] = { x: cx, y: cy };
      }
    }

    // SVG viewport dimensions
    var svgW = padX * 2 + (maxLayer + 1) * layerGapX;
    var svgH = padY * 2 + maxNodesInLayer * nodeGapY;
    if (svgW < 500) svgW = 500;
    if (svgH < 200) svgH = 200;

    svg.setAttribute('width', svgW);
    svg.setAttribute('height', svgH);
    svg.style.minWidth = svgW + 'px';

    var html = '';

    // SVG defs: arrowhead markers + glow filter
    html += '<defs>';
    html += '<marker id="dagArrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">';
    html += '<path d="M0,0 L8,3 L0,6" fill="rgba(102,204,255,0.5)" />';
    html += '</marker>';
    html += '<marker id="dagArrowDone" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">';
    html += '<path d="M0,0 L8,3 L0,6" fill="rgba(102,255,170,0.5)" />';
    html += '</marker>';
    html += '<filter id="dagGlow" x="-50%" y="-50%" width="200%" height="200%">';
    html += '<feGaussianBlur stdDeviation="3" result="blur" />';
    html += '<feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>';
    html += '</filter>';
    html += '</defs>';

    // Build a quick lookup for prerequisite status
    var taskById = {};
    for (var ti = 0; ti < tasks.length; ti++) {
      taskById[tasks[ti].id] = tasks[ti];
    }

    // ── Draw edges (prerequisite -> dependent) ──────────────────────────
    for (var i = 0; i < tasks.length; i++) {
      var task = tasks[i];
      var deps = task.depends_on || [];
      if (deps.length === 0) continue;

      var toPos = positions[task.id];
      if (!toPos) continue;

      for (var j = 0; j < deps.length; j++) {
        var fromPos = positions[deps[j]];
        if (!fromPos) continue;

        var prereq = taskById[deps[j]];
        var resolved = prereq && prereq.status === 'done';
        var edgeColor = resolved ? 'rgba(102,255,170,0.4)' : 'rgba(102,204,255,0.25)';
        var markerRef = resolved ? 'url(#dagArrowDone)' : 'url(#dagArrow)';

        // Bezier: right edge of source -> left edge of target
        var x1 = fromPos.x + nodeW / 2;
        var y1 = fromPos.y;
        var x2 = toPos.x - nodeW / 2;
        var y2 = toPos.y;
        var cpOff = Math.abs(x2 - x1) * 0.4;

        html += '<path d="M' + x1 + ',' + y1 +
                ' C' + (x1 + cpOff) + ',' + y1 +
                ' ' + (x2 - cpOff) + ',' + y2 +
                ' ' + x2 + ',' + y2 + '"' +
                ' fill="none" stroke="' + edgeColor + '"' +
                ' stroke-width="1.5" marker-end="' + markerRef + '" />';
      }
    }

    // ── Draw nodes ──────────────────────────────────────────────────────
    for (var t = 0; t < tasks.length; t++) {
      var tk = tasks[t];
      var pos = positions[tk.id];
      if (!pos) continue;

      var color = _dagColor(tk.status);
      var rx = pos.x - nodeW / 2;
      var ry = pos.y - nodeH / 2;

      // Node rectangle
      html += '<rect x="' + rx + '" y="' + ry + '"' +
              ' width="' + nodeW + '" height="' + nodeH + '"' +
              ' rx="6" ry="6"' +
              ' fill="rgba(10,15,30,0.7)"' +
              ' stroke="' + color + '"' +
              ' stroke-width="1.5" />';

      // Glow effect for in-progress nodes
      if (tk.status === 'in_progress') {
        html += '<rect x="' + rx + '" y="' + ry + '"' +
                ' width="' + nodeW + '" height="' + nodeH + '"' +
                ' rx="6" ry="6"' +
                ' fill="none" stroke="' + color + '"' +
                ' stroke-width="0.5"' +
                ' filter="url(#dagGlow)" opacity="0.6" />';
      }

      // Task ID text
      var label = tk.id || '';
      html += '<text x="' + pos.x + '" y="' + (pos.y - 3) + '"' +
              ' text-anchor="middle"' +
              ' fill="' + color + '"' +
              ' font-size="10" font-family="Courier New, monospace"' +
              ' font-weight="600">' + _svgEscape(label) + '</text>';

      // Task title (truncated to fit)
      var title = tk.title || '';
      if (title.length > 16) title = title.substring(0, 15) + '...';
      html += '<text x="' + pos.x + '" y="' + (pos.y + 10) + '"' +
              ' text-anchor="middle"' +
              ' fill="rgba(102,204,255,0.56)"' +
              ' font-size="8" font-family="Courier New, monospace">' +
              _svgEscape(title) + '</text>';

      // Status indicator dot (top-left corner)
      html += '<circle cx="' + (rx + 8) + '" cy="' + (ry + 8) + '"' +
              ' r="3" fill="' + color + '" opacity="0.8" />';
    }

    svg.innerHTML = html;
  }

  // ── Panel toggle ──────────────────────────────────────────────────────

  window.toggleDagPanel = function() {
    var overlay = document.getElementById('dagPanelOverlay');
    if (!overlay) return;
    _dagVisible = !_dagVisible;
    if (_dagVisible) {
      overlay.classList.remove('hidden');
      _renderDag();
    } else {
      overlay.classList.add('hidden');
    }
  };

  // Called from websocket.js on every status update to keep the DAG current
  window.updateDagPanel = function() {
    if (_dagVisible) _renderDag();
  };

  // ── Keyboard shortcut: Shift+D to toggle, Escape to dismiss ──────────

  document.addEventListener('keydown', function(e) {
    if (e.key === 'D' && e.shiftKey && !e.ctrlKey && !e.metaKey &&
        document.activeElement.tagName !== 'INPUT' &&
        document.activeElement.tagName !== 'TEXTAREA') {
      toggleDagPanel();
    }
  });
})();
