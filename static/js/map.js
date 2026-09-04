/**
 * Leaflet Map Controller for Multi-Sensor Dead Reckoning Visualization
 */

class NavigationMap {
    constructor(elementId) {
        this.elementId = elementId;
        this.map = null;
        
        // Path polylines
        this.gtPolyline = null;
        this.gnssPolyline = null;
        this.insPolyline = null;
        this.snappedPolyline = null;
        
        // Coordinates history
        this.gtCoords = [];
        this.gnssCoords = [];
        this.insCoords = [];
        this.snappedCoords = [];
        
        // Vehicle marker
        this.carMarker = null;
        
        this.initMap();
    }

    initMap() {
        // Default center (will auto-adjust on trip load)
        this.map = L.map(this.elementId, {
            zoomControl: true,
            attributionControl: false
        }).setView([37.7749, -122.4194], 16);

        // Dark Matter Basemap tiles (CartoDB)
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            maxZoom: 20,
            subdomains: 'abcd'
        }).addTo(this.map);

        // Ground Truth Polyline (Grey dashed)
        this.gtPolyline = L.polyline([], {
            color: '#94a3b8',
            weight: 3,
            opacity: 0.7,
            dashArray: '4, 6'
        }).addTo(this.map);

        // GNSS Raw Polyline (Blue)
        this.gnssPolyline = L.polyline([], {
            color: '#3b82f6',
            weight: 3.5,
            opacity: 0.8
        }).addTo(this.map);

        // Dead Reckoning Polyline (Orange)
        this.insPolyline = L.polyline([], {
            color: '#f59e0b',
            weight: 4,
            opacity: 0.9
        }).addTo(this.map);

        // HMM Lane-snapped Polyline (Green)
        this.snappedPolyline = L.polyline([], {
            color: '#10b981',
            weight: 4,
            opacity: 0.95
        }).addTo(this.map);

        // Custom Vehicle Navigation Marker
        const carIcon = L.divIcon({
            className: 'car-marker-container',
            html: `
                <div id="carHeadingArrow" style="transform: rotate(0deg); transform-origin: center;">
                    <svg class="w-7 h-7 car-icon" viewBox="0 0 24 24" fill="none">
                        <polygon points="12,2 22,21 12,17 2,21" fill="#3b82f6" stroke="#ffffff" stroke-width="2"/>
                    </svg>
                </div>
            `,
            iconSize: [28, 28],
            iconAnchor: [14, 14]
        });

        this.carMarker = L.marker([37.7749, -122.4194], { icon: carIcon }).addTo(this.map);
    }

    setCenter(lat, lon, zoom = 17) {
        this.map.setView([lat, lon], zoom);
    }

    clearPaths() {
        this.gtCoords = [];
        this.gnssCoords = [];
        this.insCoords = [];
        this.snappedCoords = [];
        this.gtPolyline.setLatLngs([]);
        this.gnssPolyline.setLatLngs([]);
        this.insPolyline.setLatLngs([]);
        this.snappedPolyline.setLatLngs([]);
    }

    updateState(telemetry) {
        const pos = telemetry.current_position;
        if (!pos) return;

        const gtLatLon = [pos.gt.lat, pos.gt.lon];
        const insLatLon = [pos.ins.lat, pos.ins.lon];
        const snappedLatLon = [pos.snapped.lat, pos.snapped.lon];

        // Append to history
        this.gtCoords.push(gtLatLon);
        this.insCoords.push(insLatLon);
        this.snappedCoords.push(snappedLatLon);

        if (!telemetry.blackout_active) {
            this.gnssCoords.push(gtLatLon);
        }

        // Update polylines
        this.gtPolyline.setLatLngs(this.gtCoords);
        this.insPolyline.setLatLngs(this.insCoords);
        this.snappedPolyline.setLatLngs(this.snappedCoords);
        this.gnssPolyline.setLatLngs(this.gnssCoords);

        // Update Car Marker position & rotation (prioritize snapped position for UI display)
        const displayLatLon = telemetry.blackout_active ? snappedLatLon : gtLatLon;
        this.carMarker.setLatLng(displayLatLon);
        
        const arrow = document.getElementById('carHeadingArrow');
        if (arrow) {
            arrow.style.transform = `rotate(${telemetry.heading_deg}deg)`;
        }

        // Auto pan map if vehicle gets near edges
        if (!this.map.getBounds().pad(-0.1).contains(displayLatLon)) {
            this.map.panTo(displayLatLon, { animate: true, duration: 0.3 });
        }
    }
}
