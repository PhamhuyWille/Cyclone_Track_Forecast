import './style.css'
import 'leaflet/dist/leaflet.css'
import L from 'leaflet'

// Fix Leaflet default icon path issues with bundlers
import iconRetinaUrl from 'leaflet/dist/images/marker-icon-2x.png';
import iconUrl from 'leaflet/dist/images/marker-icon.png';
import shadowUrl from 'leaflet/dist/images/marker-shadow.png';

L.Icon.Default.mergeOptions({
  iconRetinaUrl,
  iconUrl,
  shadowUrl,
});

const API_BASE = 'http://localhost:8000/api';
let pastLayer = L.layerGroup();
let map;
let markersLayer = L.layerGroup();
let linesLayer = L.layerGroup();
let currentStormTimes = [];

// Helper function for delays (animations)
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

const stormIcon = L.divIcon({
  html: '<div class="storm-spinner" style="font-size: 28px; text-shadow: 0 0 15px #00d2ff;">🌀</div>',
  className: 'custom-storm-icon',
  iconSize: [30, 30],
  iconAnchor: [15, 15]
});

// Initialize Map
function initMap() {
  map = L.map('map', {
    zoomControl: false // We will customize it
  }).setView([16.05, 108.20], 5); // Default to Da Nang

  L.control.zoom({
    position: 'bottomright'
  }).addTo(map);

  // Use Google Maps with Vietnamese localization (hl=vi)
  L.tileLayer('https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}&hl=vi', {
    attribution: '&copy; Google Maps',
    maxZoom: 20
  }).addTo(map);

  pastLayer.addTo(map);
  markersLayer.addTo(map);
  linesLayer.addTo(map);
}

// Haversine formula to compute distance in km
function haversine(lat1, lon1, lat2, lon2) {
  const R = 6371.0; 
  const dLat = (lat2 - lat1) * Math.PI / 180.0;
  const dLon = (lon2 - lon1) * Math.PI / 180.0;
  lat1 = (lat1) * Math.PI / 180.0;
  lat2 = (lat2) * Math.PI / 180.0;
  const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
            Math.sin(dLon/2) * Math.sin(dLon/2) * Math.cos(lat1) * Math.cos(lat2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a)); 
  return R * c;
}

// Fetch storms
async function loadStorms() {
  try {
    const res = await fetch(`${API_BASE}/storms`);
    const storms = await res.json();
    
    const select = document.getElementById('storm-select');
    select.innerHTML = '<option value="" disabled selected>-- Chọn một cơn bão --</option>';
    
    storms.forEach(storm => {
      const opt = document.createElement('option');
      opt.value = storm.sid;
      const dpeText = storm.dpe_24h ? ` | DPE: ${storm.dpe_24h} km` : '';
      opt.textContent = `Bão ${storm.sid} (${storm.month}/${storm.year})${dpeText}`;
      select.appendChild(opt);
    });
    
    select.addEventListener('change', (e) => loadStormTimes(e.target.value));
  } catch (err) {
    console.error(err);
    alert('Không thể tải danh sách bão. Hãy chắc chắn Backend đang chạy.');
  }
}

// Fetch times for a storm
async function loadStormTimes(sid) {
  try {
    const res = await fetch(`${API_BASE}/storms/${sid}/times`);
    currentStormTimes = await res.json();
    
    const timeSelect = document.getElementById('time-select');
    timeSelect.innerHTML = '<option value="" disabled selected>-- Chọn thời điểm khởi điểm --</option>';
    
    currentStormTimes.forEach(t => {
      const opt = document.createElement('option');
      opt.value = t.index;
      opt.textContent = t.time_str;
      timeSelect.appendChild(opt);
    });
    
    timeSelect.disabled = false;
    timeSelect.addEventListener('change', () => {
      document.getElementById('predict-btn').disabled = false;
      
      const selectedIndex = parseInt(timeSelect.value);
      
      // Find all points in the past up to selected index
      const currentIndexInArray = currentStormTimes.findIndex(t => t.index === selectedIndex);
      const point = currentStormTimes[currentIndexInArray];
      
      if(point) {
        pastLayer.clearLayers();
        markersLayer.clearLayers();
        linesLayer.clearLayers();
        document.getElementById('stats-panel').classList.add('hidden');
        
        // Draw past trajectory
        const pastPoints = currentStormTimes.slice(0, currentIndexInArray + 1).map(t => [t.lat, t.lon]);
        
        // Animated Polyline for the past track using SVG animation
        const pastLine = L.polyline(pastPoints, {
            color: '#a0aec0', 
            weight: 3, 
            dashArray: '4, 6', 
            className: 'animated-path'
        }).addTo(pastLayer);
        
        // Add start marker (Spinning Storm)
        L.marker([point.lat, point.lon], {icon: stormIcon})
          .addTo(pastLayer)
          .bindPopup(`<b>${point.time_str}</b><br/>Tâm bão hiện tại (T0)`)
          .openPopup();
        
        // Fit map bounds to past trajectory
        if (pastPoints.length > 1) {
            map.fitBounds(L.latLngBounds(pastPoints).pad(0.1));
        } else {
            map.setView([point.lat, point.lon], 6, {animate: true});
        }
      }
    });
  } catch (err) {
    console.error(err);
  }
}

async function predictTrack() {
  const btn = document.getElementById('predict-btn');
  const index = document.getElementById('time-select').value;
  if (!index) return;

  btn.disabled = true;
  btn.textContent = "Đang tính toán...";

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ index: parseInt(index) })
    });
    const data = await res.json();
    await drawPredictionsAnimated(data);
  } catch (err) {
    console.error(err);
    alert('Lỗi dự báo!');
  } finally {
    btn.disabled = false;
    btn.textContent = "Dự đoán Quỹ đạo";
  }
}

async function drawPredictionsAnimated(data) {
  markersLayer.clearLayers();
  linesLayer.clearLayers();

  const startLat = data.current_lat;
  const startLon = data.current_lon;

  let truePoints = [[startLat, startLon]];
  let predPoints = [[startLat, startLon]];
  
  const horizons = ['6h', '12h', '24h'];
  document.getElementById('stats-panel').classList.remove('hidden');

  // Draw step by step with delay (Weather forecast animation style)
  for (const h of horizons) {
    await sleep(600); // 600ms delay per step

    const hData = data.predictions[h];
    const trueLat = hData.true_lat;
    const trueLon = hData.true_lon;
    const predLat = hData.pred_lat;
    const predLon = hData.pred_lon;
    
    truePoints.push([trueLat, trueLon]);
    predPoints.push([predLat, predLon]);
    
    // Draw True point with pop effect
    L.circleMarker([trueLat, trueLon], {
      color: '#00d2ff', fillColor: '#00d2ff', fillOpacity: 0.8, radius: 6,
      className: 'marker-pop'
    }).addTo(markersLayer).bindPopup(`Thực tế ${h}`);

    // Draw Pred point with pop effect
    L.circleMarker([predLat, predLon], {
      color: '#ff416c', fillColor: '#ff416c', fillOpacity: 0.8, radius: 6,
      className: 'marker-pop'
    }).addTo(markersLayer).bindPopup(`Dự đoán ${h}`);
    
    // Draw Uncertainty Circle (Cone of Uncertainty)
    let errorRadius = 35000; // 6h default (~35km)
    if (h === '12h') errorRadius = 61000; // ~61km
    if (h === '24h') errorRadius = 145000; // ~145km
    
    L.circle([predLat, predLon], {
      color: '#ff416c',
      fillColor: '#ff416c',
      fillOpacity: 0.15,
      weight: 1.5,
      dashArray: '4, 4',
      radius: errorRadius,
      className: 'marker-pop' // Use same animation to pop in
    }).addTo(markersLayer);
    
    // Draw lines up to this point
    linesLayer.clearLayers();
    L.polyline(truePoints, {color: '#00d2ff', weight: 3, dashArray: '5, 5', className: 'pulse-line'}).addTo(linesLayer);
    L.polyline(predPoints, {color: '#ff416c', weight: 4, className: 'pulse-line'}).addTo(linesLayer);
    
    // Update Stats dynamically
    const err = haversine(trueLat, trueLon, predLat, predLon);
    document.getElementById(`stat-${h}`).textContent = `${err.toFixed(2)} km`;
    
    // Smoothly pan camera to encompass new points
    const allPoints = [...truePoints, ...predPoints];
    map.flyToBounds(L.latLngBounds(allPoints).pad(0.3), {duration: 0.5, easeLinearity: 0.25});
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initMap();
  loadStorms();
  document.getElementById('predict-btn').addEventListener('click', predictTrack);
});
