/**
 * OrbitalDelta — Leaflet Map Viewer
 * Interactive satellite change detection visualization.
 *
 * Connects to the FastAPI backend at /api/v1/ to:
 *  - Load all change detections and render as colour-coded polygons
 *  - Perform spatial bounding-box queries via drawn rectangles
 *  - Submit new image pairs for processing
 *  - Show per-polygon attributes in an info panel
 */

'use strict';

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const API_BASE = '/api/v1';
const POLL_INTERVAL_MS = 5_000; // status polling for submitted jobs

// Confidence-based colour ramp (high = red, low = yellow)
function confidenceColour(conf) {
    if (conf === null || conf === undefined) return '#94a3b8';
    if (conf >= 0.85) return '#ef4444';
    if (conf >= 0.70) return '#f97316';
    if (conf >= 0.55) return '#eab308';
    return '#84cc16';
}

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────

let map;
let changeLayer;        // L.GeoJSON layer holding all polygons
let drawLayer;         // temporary layer for bbox rectangle
let allDetections = [];  // raw GeoJSON features
let drawControl;

// ─────────────────────────────────────────────────────────────────────────────
// Map initialisation
// ─────────────────────────────────────────────────────────────────────────────

function initMap() {
    map = L.map('map', {
        center: [20, 0],
        zoom: 2,
        zoomControl: true,
    });

    // OpenStreetMap base tiles (zero-cost, no API key)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution:
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19,
    }).addTo(map);

    // Empty GeoJSON layer for change polygons
    changeLayer = L.geoJSON(null, {
        style: styleFeature,
        onEachFeature: bindPopup,
    }).addTo(map);

    // Draw layer for bbox queries
    drawLayer = new L.FeatureGroup().addTo(map);
    drawControl = new L.Control.Draw({
        draw: {
            rectangle: true,
            polygon: false,
            marker: false,
            circle: false,
            circlemarker: false,
            polyline: false,
        },
        edit: false,
    });
}

function styleFeature(feature) {
    const conf = feature.properties?.confidence || feature.properties?.mean_confidence;
    return {
        color: confidenceColour(conf),
        weight: 2,
        opacity: 0.9,
        fillOpacity: 0.35,
    };
}

function bindPopup(feature, layer) {
    layer.on('click', () => showInfo(feature));
}

// ─────────────────────────────────────────────────────────────────────────────
// API helpers
// ─────────────────────────────────────────────────────────────────────────────

async function apiFetch(path, options = {}) {
    const resp = await fetch(API_BASE + path, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
    });
    if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`API ${path} → ${resp.status}: ${text}`);
    }
    return resp.json();
}

// ─────────────────────────────────────────────────────────────────────────────
// Load & render detections
// ─────────────────────────────────────────────────────────────────────────────

async function loadDetections() {
    try {
        const data = await apiFetch('/detections');
        allDetections = Array.isArray(data) ? data : (data.features || []);
        renderDetections(allDetections);
    } catch (err) {
        console.warn('Could not load detections:', err.message);
        renderDetections([]);
    }
}

function renderDetections(detections) {
    changeLayer.clearLayers();

    if (!detections.length) {
        updateStats([]);
        updateList([]);
        return;
    }

    // Support both raw feature arrays and full FeatureCollection
    const features = detections.map((d) => {
        if (d.type === 'Feature') return d;
        // Wrap plain detection objects
        return {
            type: 'Feature',
            geometry: typeof d.geometry === 'string' ? JSON.parse(d.geometry) : d.geometry,
            properties: d,
        };
    }).filter((f) => f.geometry);

    const fc = { type: 'FeatureCollection', features };
    changeLayer.addData(fc);

    // Zoom to results if we have any and map is zoomed out
    if (features.length && map.getZoom() < 4) {
        try {
            map.fitBounds(changeLayer.getBounds(), { padding: [40, 40] });
        } catch (_) { /* empty layer */ }
    }

    updateStats(features);
    updateList(features);

    // Apply current area filter
    applyAreaFilter();
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics sidebar
// ─────────────────────────────────────────────────────────────────────────────

function updateStats(features) {
    const total = features.length;
    const totalArea = features.reduce((s, f) => s + (f.properties?.area_m2 || 0), 0);
    const confs = features.map((f) => f.properties?.confidence || f.properties?.mean_confidence || 0).filter(Boolean);
    const avgConf = confs.length ? confs.reduce((a, b) => a + b, 0) / confs.length : null;

    document.getElementById('stat-total').textContent = total;
    document.getElementById('stat-area').textContent = total ? totalArea.toFixed(0) : '—';
    document.getElementById('stat-conf').textContent = avgConf !== null ? avgConf.toFixed(3) : '—';
}

function updateList(features) {
    const list = document.getElementById('detection-list');
    const badge = document.getElementById('list-count');
    badge.textContent = features.length;

    list.innerHTML = features.slice(0, 100).map((f, i) => {
        const p = f.properties || {};
        const conf = (p.confidence || p.mean_confidence || 0).toFixed(3);
        const area = p.area_m2 ? p.area_m2.toFixed(0) + ' m²' : '—';
        const colour = confidenceColour(p.confidence || p.mean_confidence);
        return `
      <li class="detection-item" data-idx="${i}" style="border-left: 4px solid ${colour}">
        <div class="det-id">#${p.detection_id?.slice(0, 8) || i}</div>
        <div class="det-meta">Area: ${area} &bull; Conf: ${conf}</div>
      </li>`;
    }).join('');

    list.querySelectorAll('.detection-item').forEach((el) => {
        el.addEventListener('click', () => {
            const idx = parseInt(el.dataset.idx, 10);
            const feature = features[idx];
            if (!feature) return;
            try {
                const bounds = L.geoJSON(feature).getBounds();
                map.fitBounds(bounds, { padding: [80, 80] });
            } catch (_) { /* no geometry */ }
            showInfo(feature);
        });
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Info panel
// ─────────────────────────────────────────────────────────────────────────────

function showInfo(feature) {
    const panel = document.getElementById('info-panel');
    const table = document.getElementById('info-table');
    panel.classList.remove('hidden');

    const p = feature.properties || {};
    const rows = [
        ['Detection ID', p.detection_id || '—'],
        ['Area', p.area_m2 ? p.area_m2.toFixed(2) + ' m²' : '—'],
        ['Confidence', (p.confidence || p.mean_confidence || 0).toFixed(4)],
        ['Timestamp A', p.timestamp_a || '—'],
        ['Timestamp B', p.timestamp_b || '—'],
        ['Centroid X', typeof p.centroid_x !== 'undefined' ? p.centroid_x?.toFixed(6) : '—'],
        ['Centroid Y', typeof p.centroid_y !== 'undefined' ? p.centroid_y?.toFixed(6) : '—'],
        ['Created at', p.created_at || '—'],
    ];

    table.innerHTML = rows.map(([k, v]) =>
        `<tr><th>${k}</th><td>${v}</td></tr>`
    ).join('');
}

document.getElementById('info-close').addEventListener('click', () => {
    document.getElementById('info-panel').classList.add('hidden');
});

// ─────────────────────────────────────────────────────────────────────────────
// Area filter slider
// ─────────────────────────────────────────────────────────────────────────────

const areaSlider = document.getElementById('area-filter');
const areaVal = document.getElementById('area-val');

areaSlider.addEventListener('input', () => {
    areaVal.textContent = areaSlider.value;
    applyAreaFilter();
});

function applyAreaFilter() {
    const minArea = parseInt(areaSlider.value, 10);
    changeLayer.eachLayer((layer) => {
        const area = layer.feature?.properties?.area_m2 || 0;
        if (area >= minArea) {
            layer.setStyle({ opacity: 0.9, fillOpacity: 0.35 });
        } else {
            layer.setStyle({ opacity: 0, fillOpacity: 0 });
        }
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer toggles
// ─────────────────────────────────────────────────────────────────────────────

document.getElementById('chk-changes').addEventListener('change', (e) => {
    if (e.target.checked) map.addLayer(changeLayer);
    else map.removeLayer(changeLayer);
});

// ─────────────────────────────────────────────────────────────────────────────
// Bounding-box spatial query
// ─────────────────────────────────────────────────────────────────────────────

document.getElementById('btn-query-bbox').addEventListener('click', () => {
    // Activate Leaflet Draw rectangle tool
    map.addControl(drawControl);
    const rectHandler = new L.Draw.Rectangle(map, drawControl.options.draw.rectangle);
    rectHandler.enable();
});

map.on(L.Draw.Event.CREATED, async (e) => {
    drawLayer.clearLayers();
    drawLayer.addLayer(e.layer);
    map.removeControl(drawControl);

    const bounds = e.layer.getBounds();
    const bbox = {
        xmin: bounds.getWest(),
        ymin: bounds.getSouth(),
        xmax: bounds.getEast(),
        ymax: bounds.getNorth(),
    };

    try {
        const results = await apiFetch('/detections/query', {
            method: 'POST',
            body: JSON.stringify(bbox),
        });
        renderDetections(Array.isArray(results) ? results : (results.features || []));
    } catch (err) {
        console.error('Bbox query failed:', err.message);
    }
});

// ─────────────────────────────────────────────────────────────────────────────
// Refresh button
// ─────────────────────────────────────────────────────────────────────────────

document.getElementById('btn-refresh').addEventListener('click', () => {
    drawLayer.clearLayers();
    loadDetections();
});

// ─────────────────────────────────────────────────────────────────────────────
// Submit modal
// ─────────────────────────────────────────────────────────────────────────────

const modal = document.getElementById('modal-backdrop');
const jobStatus = document.getElementById('job-status');

document.getElementById('btn-submit').addEventListener('click', () => {
    modal.classList.remove('hidden');
    jobStatus.classList.add('hidden');
    jobStatus.textContent = '';
});

document.getElementById('modal-close').addEventListener('click', closeModal);
document.getElementById('btn-cancel').addEventListener('click', closeModal);

modal.addEventListener('click', (e) => {
    if (e.target === modal) closeModal();
});

function closeModal() {
    modal.classList.add('hidden');
}

document.getElementById('btn-detect').addEventListener('click', async () => {
    const imgA = document.getElementById('in-img-a').value.trim();
    const imgB = document.getElementById('in-img-b').value.trim();
    const ckpt = document.getElementById('in-ckpt').value.trim();
    const tsA = document.getElementById('in-ts-a').value.trim();
    const tsB = document.getElementById('in-ts-b').value.trim();

    if (!imgA || !imgB) {
        showJobStatus('⚠️ Both image paths are required.', 'warning');
        return;
    }

    showJobStatus('⏳ Submitting job…', 'info');

    try {
        const payload = {
            img_a_path: imgA,
            img_b_path: imgB,
        };
        if (ckpt) payload.checkpoint_path = ckpt;
        if (tsA) payload.timestamp_a = tsA;
        if (tsB) payload.timestamp_b = tsB;

        const resp = await apiFetch('/detect', { method: 'POST', body: JSON.stringify(payload) });
        const jobId = resp.job_id;
        showJobStatus(`✅ Job submitted (ID: ${jobId?.slice(0, 8) || 'N/A'}). Polling for results…`, 'success');

        // Poll for job completion
        if (jobId) pollJob(jobId);
    } catch (err) {
        showJobStatus(`❌ Error: ${err.message}`, 'error');
    }
});

function showJobStatus(msg, type = 'info') {
    jobStatus.className = `job-status job-status--${type}`;
    jobStatus.textContent = msg;
    jobStatus.classList.remove('hidden');
}

async function pollJob(jobId) {
    const maxAttempts = 60; // 5 minutes max
    let attempts = 0;

    const poll = async () => {
        attempts++;
        if (attempts > maxAttempts) {
            showJobStatus('⚠️ Timed out waiting for job. Check back later.', 'warning');
            return;
        }

        try {
            const status = await apiFetch(`/jobs/${jobId}`);
            if (status.status === 'completed') {
                showJobStatus(`✅ Detection complete! Refreshing map…`, 'success');
                setTimeout(() => {
                    closeModal();
                    loadDetections();
                }, 1500);
            } else if (status.status === 'failed') {
                showJobStatus(`❌ Job failed: ${status.error || 'unknown error'}`, 'error');
            } else {
                showJobStatus(`⏳ Status: ${status.status}… (${attempts}/${maxAttempts})`, 'info');
                setTimeout(poll, POLL_INTERVAL_MS);
            }
        } catch (err) {
            console.warn('Poll error:', err.message);
            setTimeout(poll, POLL_INTERVAL_MS);
        }
    };

    setTimeout(poll, POLL_INTERVAL_MS);
}

// ─────────────────────────────────────────────────────────────────────────────
// Boot
// ─────────────────────────────────────────────────────────────────────────────

initMap();
loadDetections();

// Auto-refresh every 30 seconds
setInterval(loadDetections, 30_000);
