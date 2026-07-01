"use strict";

const $ = (id) => document.getElementById(id);

let report = null;
let selectedCheckpointIndex = 0;

function fmt(value, digits = 3) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : String(value ?? "-");
}

function num(id) {
  return Number($(id).value);
}

function normalizedY(index, count) {
  return count === 1 ? 0.5 : index / (count - 1);
}

function setHidden(id, hidden) {
  $(id).classList.toggle("hidden", hidden);
}

function collectPayload() {
  const mode = $("mode").value;
  const common = {
    mode,
    epochs: num("epochs"),
    seedCount: num("seedCount"),
    axonThreshold: num("axonThreshold"),
    branchThreshold: num("branchThreshold")
  };

  if (mode === "valence") {
    return {
      ...common,
      mediumCount: num("valenceMediumCount"),
      stemFanout: num("stemFanout"),
      readoutFast: num("readoutFast"),
      trialsPerEpoch: num("trialsPerEpoch"),
      taggedMode: $("taggedMode").value,
      globalIncrement: num("globalIncrement"),
      tagObjectStimWithToxin: $("tagObjectStimWithToxin").checked
    };
  }

  if (mode === "gridWorld") {
    return {
      ...common,
      objectSensorCount: num("objectSensorCount"),
      mediumRows: num("mediumRows"),
      mediumCols: num("mediumCols"),
      slotsPerNeuron: num("gridSlotsPerNeuron"),
      fastInit: num("gridFastInit"),
      candidateMaxAge: num("candidateMaxAge"),
      dormantLimit: num("dormantLimit"),
      taggedMode: $("gridTaggedMode").value,
      globalIncrement: num("gridGlobalIncrement"),
      tagObjectStimWithToxin: $("gridTagObjectStimWithToxin").checked
    };
  }

  return {
    ...common,
    inputCount: num("inputCount"),
    mediumCount: num("mediumCount"),
    outputCount: num("outputCount"),
    slotsPerNeuron: num("slotsPerNeuron"),
    scale: num("scale"),
    fastInit: num("fastInit"),
    candidateMaxAge: num("candidateMaxAge"),
    dormantLimit: num("dormantLimit")
  };
}

function syncModeControls() {
  const valence = $("mode").value === "valence";
  const gridWorld = $("mode").value === "gridWorld";
  setHidden("naturalControls", valence || gridWorld);
  setHidden("gridWorldControls", !gridWorld);
  setHidden("valenceControls", !valence);
  setHidden("trialsPerEpochRow", !valence);
  setHidden("candidateMaxAgeRow", valence);
  setHidden("dormantLimitRow", valence);
  setHidden("naturalInspector", valence || gridWorld);
  setHidden("valenceInspector", !valence);
  setHidden("gridWorldInspector", !gridWorld);
  if (valence && Number($("epochs").value) < 300) $("epochs").value = 300;
  if (gridWorld && Number($("epochs").value) < 300) $("epochs").value = 300;
}

async function runSimulation() {
  const button = $("runButton");
  button.disabled = true;
  button.textContent = "Running";
  try {
    const response = await fetch("/api/simulate", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(collectPayload())
    });
    const body = await response.json();
    if (!response.ok || body.error) {
      throw new Error(body.error || `HTTP ${response.status}`);
    }
    report = body;
    selectedCheckpointIndex = Math.max(0, report.checkpoints.length - 1);
    updateCheckpointSlider();
    populateSelectors();
    renderAll();
  } catch (error) {
    $("summary").innerHTML = `<div class="metric"><b>error</b><span>${escapeHtml(error.message)}</span></div>`;
  } finally {
    button.disabled = false;
    button.textContent = "Run";
  }
}

function updateCheckpointSlider() {
  const slider = $("checkpointSlider");
  const max = report ? Math.max(0, report.checkpoints.length - 1) : 0;
  slider.max = String(max);
  slider.value = String(selectedCheckpointIndex);
}

function checkpoint() {
  if (!report || report.checkpoints.length === 0) return null;
  return report.checkpoints[selectedCheckpointIndex] ?? report.checkpoints[report.checkpoints.length - 1];
}

function populateSelectors() {
  const cp = checkpoint();
  if (!cp) return;

  if (report.mode === "natural") {
    const current = $("inputSelector").value;
    $("inputSelector").innerHTML = cp.paths
      .map((path) => `<option value="${path.inputId}">${path.inputId} -> ${path.targetOutputId}</option>`)
      .join("");
    if (current && cp.paths.some((path) => path.inputId === current)) $("inputSelector").value = current;
  } else if (report.mode === "gridWorld") {
    const current = $("gridCaseSelector").value;
    $("gridCaseSelector").innerHTML = cp.cases
      .map((item) => `<option value="${item.label}">${item.label}</option>`)
      .join("");
    if (current && cp.cases.some((item) => item.label === current)) $("gridCaseSelector").value = current;
  } else {
    const current = $("caseSelector").value;
    $("caseSelector").innerHTML = cp.cases
      .map((item) => `<option value="${item.label}">${item.label}</option>`)
      .join("");
    if (current && cp.cases.some((item) => item.label === current)) $("caseSelector").value = current;
  }
}

function renderAll() {
  const cp = checkpoint();
  if (!report || !cp) {
    drawEmpty($("worldCanvas"), "No run yet");
    drawEmpty($("networkCanvas"), "No run yet");
    drawEmpty($("curveCanvas"), "No run yet");
    return;
  }
  $("checkpointLabel").textContent = `epoch ${cp.epoch}`;
  renderSummary(cp);
  drawCurve();
  drawWorld(cp);
  drawNetwork(cp);
  renderInspector(cp);
  renderEdges(cp);
}

function renderSummary(cp) {
  const metrics = cp.metrics;
  const stats = cp.stats;
  const entries = report.mode === "natural"
    ? [
        ["SR", fmt(metrics.sr)],
        ["noop", fmt(metrics.noop)],
        ["conflict", fmt(metrics.conflict)],
        ["wrong", fmt(metrics.wrong)],
        ["stem/readout", `${fmt(stats.stem, 1)} / ${fmt(stats.readout, 1)}`],
        ["stable/pruned", `${fmt(stats.stable, 1)} / ${fmt(stats.pruned, 1)}`]
      ]
    : report.mode === "gridWorld"
      ? [
          ["correct", fmt(metrics.sr)],
          ["noop", fmt(metrics.noop)],
          ["conflict", fmt(metrics.conflict)],
          ["wrong", fmt(metrics.wrong)],
          ["left/right", `${fmt(metrics.left)} / ${fmt(metrics.right)}`],
          ["stable/pruned", `${fmt(stats.stable, 1)} / ${fmt(stats.pruned, 1)}`]
        ]
    : [
        ["nutrient", fmt(metrics.nutrient)],
        ["toxin", fmt(metrics.toxin)],
        ["noop", fmt(metrics.noop)],
        ["conflict", fmt(metrics.conflict)],
        ["toxinD", fmt(metrics.toxinDrive)],
        ["nutrD", fmt(metrics.nutrientDrive)]
      ];

  $("summary").innerHTML = entries
    .map(([label, value]) => `<div class="metric"><b>${escapeHtml(label)}</b><span>${escapeHtml(value)}</span></div>`)
    .join("");
}

function selectedNaturalPath(cp) {
  const selected = $("inputSelector").value;
  return cp.paths.find((path) => path.inputId === selected) ?? cp.paths[0] ?? null;
}

function selectedValenceCase(cp) {
  const selected = $("caseSelector").value;
  return cp.cases.find((item) => item.label === selected) ?? cp.cases[0] ?? null;
}

function selectedGridWorldCase(cp) {
  const selected = $("gridCaseSelector").value;
  return cp.cases.find((item) => item.label === selected) ?? cp.cases[0] ?? null;
}

function renderInspector(cp) {
  if (report.mode === "natural") {
    const path = selectedNaturalPath(cp);
    if (!path) return;
    $("pathDetails").innerHTML = kvHtml([
      ["action", path.action],
      ["outputs", path.activeOutputs.join("|") || "-"],
      ["stemN", fmt(path.liveStemCount, 1)],
      ["fireN", fmt(path.firingMediumCount, 1)],
      ["stemEff", fmt(path.liveStemEff, 2)],
      ["actStem", fmt(path.activeStemEff, 2)],
      ["correctD", fmt(path.correctDrive, 2)],
      ["wrongD", fmt(path.wrongMaxDrive, 2)]
    ]);
    $("weightTable").innerHTML = tableHtml(
      ["output", "live", "eff", "fast", "stable"],
      cp.weights.map((row) => [row.outputId, fmt(row.live, 1), fmt(row.eff, 2), fmt(row.fast, 2), fmt(row.stable, 2)])
    );
  } else if (report.mode === "gridWorld") {
    const item = selectedGridWorldCase(cp);
    if (!item) return;
    $("gridCaseDetails").innerHTML = kvHtml([
      ["action", item.action],
      ["target", item.targetOutputId ?? "-"],
      ["inputs", item.activeInputs.join(", ")],
      ["outputs", item.activeOutputs.join("|") || "-"],
      ["fireN", fmt(item.firingMediumCount, 1)],
      ["stemEff", fmt(item.liveStemEff, 2)],
      ["leftD", fmt(item.leftDrive, 2)],
      ["rightD", fmt(item.rightDrive, 2)]
    ]);
    $("gridWeightTable").innerHTML = tableHtml(
      ["motor", "live", "eff", "fast", "stable"],
      cp.weights.map((row) => [row.outputId, fmt(row.live, 1), fmt(row.eff, 2), fmt(row.fast, 2), fmt(row.stable, 2)])
    );
  } else {
    const item = selectedValenceCase(cp);
    if (!item) return;
    $("caseDetails").innerHTML = kvHtml([
      ["action", item.action],
      ["inputs", item.activeInputs.join(", ")],
      ["outputs", item.activeOutputs.join("|") || "-"],
      ["fireN", fmt(item.firingMediumCount, 1)],
      ["stemEff", fmt(item.liveStemEff, 2)],
      ["actStem", fmt(item.activeStemEff, 2)],
      ["toxinD", fmt(item.toxinDrive, 2)],
      ["nutrD", fmt(item.nutrientDrive, 2)]
    ]);
  }
}

function renderEdges(cp) {
  const edges = [...(cp.graph?.edges ?? [])]
    .sort((a, b) => {
      const stateRank = { stable: 0, active: 1, candidate: 2, dormant: 3, pruned: 4 };
      return (stateRank[a.state] ?? 9) - (stateRank[b.state] ?? 9) ||
        Math.abs(b.effectiveWeight) - Math.abs(a.effectiveWeight) ||
        a.id.localeCompare(b.id);
    })
    .slice(0, 80);
  $("edgeTable").innerHTML = tableHtml(
    ["edge", "state", "eff", "fast", "stable", "use", "tag"],
    edges.map((edge) => [
      `${edge.pre}->${edge.post}`,
      edge.state,
      fmt(edge.effectiveWeight, 2),
      fmt(edge.fastWeight, 2),
      fmt(edge.stableWeight, 2),
      fmt(edge.recentUse, 2),
      fmt(edge.tagLoad, 2)
    ])
  );
}

function kvHtml(rows) {
  return rows.map(([key, value]) => `<div><b>${escapeHtml(key)}</b><span>${escapeHtml(value)}</span></div>`).join("");
}

function tableHtml(headers, rows) {
  const head = headers.map((item) => `<th>${escapeHtml(item)}</th>`).join("");
  const body = rows
    .map((row) => `<tr>${row.map((item) => `<td>${escapeHtml(item)}</td>`).join("")}</tr>`)
    .join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height || canvas.height));
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

function clearCanvas(canvas) {
  const setup = setupCanvas(canvas);
  setup.ctx.clearRect(0, 0, setup.width, setup.height);
  return setup;
}

function drawEmpty(canvas, text) {
  const { ctx, width, height } = clearCanvas(canvas);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "#65727d";
  ctx.font = "14px sans-serif";
  ctx.fillText(text, 18, 30);
}

function drawCurve() {
  const canvas = $("curveCanvas");
  const { ctx, width, height } = clearCanvas(canvas);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  if (!report) return;

  const pad = { left: 34, right: 12, top: 14, bottom: 26 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const rows = report.checkpoints;
  const maxEpoch = rows[rows.length - 1]?.epoch ?? 1;

  ctx.strokeStyle = "#d5dce2";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();

  const series = report.mode === "gridWorld"
    ? [
        ["correct", "#2f8f5b", (row) => row.metrics.sr],
        ["left", "#2878a8", (row) => row.metrics.left],
        ["right", "#b7791f", (row) => row.metrics.right],
        ["conflict", "#b94a48", (row) => row.metrics.conflict],
        ["noop", "#65727d", (row) => row.metrics.noop]
      ]
    : report.mode === "natural"
    ? [
        ["correct", "#2f8f5b", (row) => row.metrics.sr],
        ["noop", "#65727d", (row) => row.metrics.noop],
        ["conflict", "#b94a48", (row) => row.metrics.conflict],
        ["wrong", "#b7791f", (row) => row.metrics.wrong]
      ]
    : [
        ["nutrient", "#2f8f5b", (row) => row.metrics.nutrient],
        ["toxin", "#b7791f", (row) => row.metrics.toxin],
        ["conflict", "#b94a48", (row) => row.metrics.conflict],
        ["noop", "#65727d", (row) => row.metrics.noop]
      ];

  for (const [, color, pick] of series) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    rows.forEach((row, index) => {
      const x = pad.left + (row.epoch / maxEpoch) * plotW;
      const y = pad.top + (1 - pick(row)) * plotH;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  const selected = checkpoint();
  if (selected) {
    const x = pad.left + (selected.epoch / maxEpoch) * plotW;
    ctx.strokeStyle = "#1d252c";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, pad.top + plotH);
    ctx.stroke();
  }

  ctx.font = "11px sans-serif";
  let lx = pad.left;
  for (const [label, color] of series) {
    ctx.fillStyle = color;
    ctx.fillRect(lx, height - 16, 10, 3);
    ctx.fillStyle = "#65727d";
    ctx.fillText(label, lx + 14, height - 11);
    lx += Math.max(58, label.length * 8 + 24);
  }
}

function drawWorld(cp) {
  if (report.mode === "natural") {
    drawNaturalWorld(cp);
  } else if (report.mode === "gridWorld") {
    drawGridWorld(cp);
  } else {
    drawValenceWorld(cp);
  }
}

function drawNaturalWorld(cp) {
  const { ctx, width, height } = clearCanvas($("worldCanvas"));
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  const path = selectedNaturalPath(cp);
  const params = report.params;
  const margin = 52;
  const top = 36;
  const bottom = height - 34;

  drawAxis(ctx, margin, top, margin, bottom, "inputs");
  drawAxis(ctx, width - margin, top, width - margin, bottom, "outputs");

  for (let i = 0; i < params.inputCount; i += 1) {
    const y = top + normalizedY(i, params.inputCount) * (bottom - top);
    drawDot(ctx, margin, y, 4, path?.inputIndex === i ? "#b7791f" : "#d5dce2", "#ffffff");
  }
  for (let i = 0; i < params.outputCount; i += 1) {
    const y = top + normalizedY(i, params.outputCount) * (bottom - top);
    drawDot(ctx, width - margin, y, 4, path?.targetOutputIndex === i ? "#2f8f5b" : "#d5dce2", "#ffffff");
  }

  if (path) {
    const y1 = top + normalizedY(path.inputIndex, params.inputCount) * (bottom - top);
    const y2 = top + normalizedY(path.targetOutputIndex, params.outputCount) * (bottom - top);
    ctx.strokeStyle = path.action === "correct" ? "#2f8f5b" : path.action === "conflict" ? "#b94a48" : "#65727d";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(margin + 8, y1);
    ctx.lineTo(width - margin - 8, y2);
    ctx.stroke();
    ctx.fillStyle = "#1d252c";
    ctx.font = "13px sans-serif";
    ctx.fillText(`${path.inputId} -> ${path.targetOutputId} / ${path.action}`, margin + 18, 24);
  }
}

function drawValenceWorld(cp) {
  const { ctx, width, height } = clearCanvas($("worldCanvas"));
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  const item = selectedValenceCase(cp);
  const active = new Set(item?.activeInputs ?? []);
  const y = height / 2;
  const centerX = width / 2;

  ctx.strokeStyle = "#d5dce2";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(50, y);
  ctx.lineTo(width - 50, y);
  ctx.stroke();

  drawGate(ctx, 92, y, "toxinGate", active.has("toxinGate"), "#b7791f");
  drawGate(ctx, centerX - 90, y, "objectStimA", active.has("objectStimA"), "#2878a8");
  drawGate(ctx, centerX + 90, y, "objectStimB", active.has("objectStimB"), "#2878a8");
  drawGate(ctx, width - 92, y, "nutrientGate", active.has("nutrientGate"), "#2f8f5b");

  ctx.fillStyle = "#1d252c";
  ctx.font = "13px sans-serif";
  ctx.fillText(`${item?.label ?? "-"} / action=${item?.action ?? "-"}`, 18, 24);
  ctx.fillStyle = "#65727d";
  ctx.fillText(
    `taggedMode=${report.params.taggedMode} / tagObjectStimWithToxin=${report.params.tagObjectStimWithToxin ? "1" : "0"}`,
    18,
    height - 18
  );
}

function drawGridWorld(cp) {
  const { ctx, width, height } = clearCanvas($("worldCanvas"));
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  const item = selectedGridWorldCase(cp);
  const world = item?.world ?? { agent: 0.5, nutrient: 0.18, toxin: 0.82, object: 0.5 };
  const x0 = 62;
  const x1 = width - 62;
  const y = height / 2;
  const pos = (value) => x0 + value * (x1 - x0);

  ctx.strokeStyle = "#84909a";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x0, y);
  ctx.lineTo(x1, y);
  ctx.stroke();

  drawWorldMarker(ctx, pos(world.nutrient), y, "nutrient", "#2f8f5b");
  drawWorldMarker(ctx, pos(world.toxin), y, "toxin", "#b94a48");
  drawWorldMarker(ctx, pos(world.object), y - 30, "object", "#2878a8");
  drawAgent(ctx, pos(world.agent), y);

  ctx.fillStyle = "#1d252c";
  ctx.font = "13px sans-serif";
  ctx.fillText(`${item?.label ?? "-"} / action=${item?.action ?? "-"} / target=${item?.targetOutputId ?? "-"}`, 18, 24);
  ctx.fillStyle = "#65727d";
  ctx.fillText(`leftD=${fmt(item?.leftDrive ?? 0, 2)} rightD=${fmt(item?.rightDrive ?? 0, 2)}`, 18, height - 18);
}

function drawNetwork(cp) {
  const graph = cp.graph;
  if (!graph) {
    drawEmpty($("networkCanvas"), "No graph snapshot");
    return;
  }
  const { ctx, width, height } = clearCanvas($("networkCanvas"));
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const active = activeContext(cp);
  const coords = nodeCoordinates(graph.nodes, width, height);
  const highlightedNodes = new Set([...active.inputs, ...active.mediums, ...active.outputs]);

  for (const edge of graph.edges) {
    const a = coords.get(edge.pre);
    const b = coords.get(edge.post);
    if (!a || !b) continue;
    const highlighted = (active.inputs.has(edge.pre) && active.mediums.has(edge.post)) ||
      (active.mediums.has(edge.pre) && active.outputs.has(edge.post));
    drawEdge(ctx, a, b, edge, highlighted);
  }

  for (const node of graph.nodes) {
    const c = coords.get(node.id);
    if (!c) continue;
    drawNode(ctx, c.x, c.y, node, highlightedNodes.has(node.id));
  }

  drawLegend(ctx, width, height);
}

function activeContext(cp) {
  if (report.mode === "natural") {
    const path = selectedNaturalPath(cp);
    return {
      inputs: new Set(path ? [path.inputId] : []),
      mediums: new Set(path?.firingMediumIds ?? []),
      outputs: new Set(path?.activeOutputs ?? [])
    };
  }
  if (report.mode === "gridWorld") {
    const item = selectedGridWorldCase(cp);
    return {
      inputs: new Set(item?.activeInputs ?? []),
      mediums: new Set(item?.firingMediumIds ?? []),
      outputs: new Set(item?.activeOutputs ?? [])
    };
  }
  const item = selectedValenceCase(cp);
  return {
    inputs: new Set(item?.activeInputs ?? []),
    mediums: new Set(item?.firingMediumIds ?? []),
    outputs: new Set(item?.activeOutputs ?? [])
  };
}

function nodeCoordinates(nodes, width, height) {
  const marginX = 86;
  const marginY = 34;
  const byId = new Map();
  const xs = nodes.map((node) => Number(node.x)).filter(Number.isFinite);
  const ys = nodes.map((node) => Number(node.y)).filter(Number.isFinite);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = Math.max(0.0001, maxX - minX);
  const spanY = Math.max(0.0001, maxY - minY);
  for (const node of nodes) {
    const x = marginX + ((Number(node.x) - minX) / spanX) * (width - marginX * 2);
    const y = marginY + ((Number(node.y) - minY) / spanY) * (height - marginY * 2);
    byId.set(node.id, { x, y });
  }
  return byId;
}

function drawEdge(ctx, a, b, edge, highlighted) {
  const stateColor = {
    stable: "#2878a8",
    active: "#b7791f",
    candidate: "#84909a",
    dormant: "#aeb7bf",
    pruned: "#d9dee3"
  }[edge.state] || "#84909a";
  ctx.save();
  ctx.strokeStyle = highlighted ? "#2f8f5b" : stateColor;
  ctx.globalAlpha = edge.state === "pruned" ? 0.35 : edge.state === "dormant" ? 0.55 : 0.88;
  ctx.lineWidth = highlighted ? 3 : Math.max(0.7, Math.min(4, 0.7 + Math.abs(edge.effectiveWeight) * 1.15));
  if (edge.state === "dormant" || edge.state === "pruned") ctx.setLineDash([4, 5]);
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  const midX = (a.x + b.x) / 2;
  ctx.bezierCurveTo(midX, a.y, midX, b.y, b.x, b.y);
  ctx.stroke();
  ctx.restore();
}

function drawNode(ctx, x, y, node, highlighted) {
  const fill = node.role === "sensory" ? "#f4d28b" : node.role === "motor" ? "#9fd3b2" : "#b7c9d9";
  ctx.beginPath();
  ctx.arc(x, y, highlighted ? 8 : 6, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = highlighted ? 3 : 1;
  ctx.strokeStyle = highlighted ? "#1f6f4a" : "#52616c";
  ctx.stroke();

  ctx.fillStyle = "#1d252c";
  ctx.font = "11px sans-serif";
  const text = compactId(node.id);
  const offset = node.role === "motor" ? 10 : -10 - Math.min(36, text.length * 3);
  ctx.fillText(text, x + offset, y - 10);
}

function drawLegend(ctx, width, height) {
  const items = [
    ["stable", "#2878a8"],
    ["active", "#b7791f"],
    ["candidate", "#84909a"],
    ["dormant/pruned", "#aeb7bf"],
    ["selected path", "#2f8f5b"]
  ];
  let x = 16;
  const y = height - 16;
  ctx.font = "11px sans-serif";
  for (const [label, color] of items) {
    ctx.strokeStyle = color;
    ctx.lineWidth = label === "selected path" ? 3 : 2;
    ctx.beginPath();
    ctx.moveTo(x, y - 4);
    ctx.lineTo(x + 16, y - 4);
    ctx.stroke();
    ctx.fillStyle = "#65727d";
    ctx.fillText(label, x + 22, y);
    x += Math.max(90, label.length * 7 + 34);
  }
}

function drawAxis(ctx, x1, y1, x2, y2, label) {
  ctx.strokeStyle = "#84909a";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.fillStyle = "#65727d";
  ctx.font = "12px sans-serif";
  ctx.fillText(label, x1 - 22, y1 - 12);
}

function drawDot(ctx, x, y, r, fill, stroke) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = stroke;
  ctx.stroke();
}

function drawGate(ctx, x, y, label, active, color) {
  drawDot(ctx, x, y, active ? 18 : 14, active ? color : "#eef2f5", "#52616c");
  ctx.fillStyle = active ? "#ffffff" : "#1d252c";
  ctx.font = "12px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(active ? "on" : "off", x, y + 4);
  ctx.fillStyle = "#65727d";
  ctx.fillText(label, x, y + 36);
  ctx.textAlign = "left";
}

function drawWorldMarker(ctx, x, y, label, color) {
  drawDot(ctx, x, y, 9, color, "#ffffff");
  ctx.fillStyle = "#65727d";
  ctx.font = "12px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(label, x, y + 25);
  ctx.textAlign = "left";
}

function drawAgent(ctx, x, y) {
  ctx.strokeStyle = "#1d252c";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(x, y, 14, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x - 26, y);
  ctx.lineTo(x - 48, y);
  ctx.moveTo(x - 48, y);
  ctx.lineTo(x - 40, y - 6);
  ctx.moveTo(x - 48, y);
  ctx.lineTo(x - 40, y + 6);
  ctx.moveTo(x + 26, y);
  ctx.lineTo(x + 48, y);
  ctx.moveTo(x + 48, y);
  ctx.lineTo(x + 40, y - 6);
  ctx.moveTo(x + 48, y);
  ctx.lineTo(x + 40, y + 6);
  ctx.stroke();
  ctx.fillStyle = "#1d252c";
  ctx.font = "12px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("there", x, y - 23);
  ctx.textAlign = "left";
}

function compactId(id) {
  return String(id)
    .replace("medium", "m")
    .replace("input", "i")
    .replace("output", "o")
    .replace("objectStim", "obj")
    .replace("nutrient", "nutr")
    .replace("toxin", "tox");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

window.addEventListener("resize", () => renderAll());
$("mode").addEventListener("change", () => {
  syncModeControls();
  renderAll();
});
$("runButton").addEventListener("click", runSimulation);
$("checkpointSlider").addEventListener("input", () => {
  selectedCheckpointIndex = Number($("checkpointSlider").value);
  populateSelectors();
  renderAll();
});
$("inputSelector").addEventListener("change", renderAll);
$("caseSelector").addEventListener("change", renderAll);
$("gridCaseSelector").addEventListener("change", renderAll);

syncModeControls();
runSimulation();
