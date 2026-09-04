/**
 * Main Application Orchestrator
 */

document.addEventListener('DOMContentLoaded', () => {
    const navMap = new NavigationMap('map');
    const imuVis = new IMUVisualizer('imuChart');
    
    // UI Elements
    const tripSelect = document.getElementById('tripSelect');
    const loadTripBtn = document.getElementById('loadTripBtn');
    const playBtn = document.getElementById('playBtn');
    const resetBtn = document.getElementById('resetBtn');
    const blackoutBtn = document.getElementById('blackoutBtn');
    const statusBadge = document.getElementById('statusBadge');
    const statusText = document.getElementById('statusText');
    const blackoutAlert = document.getElementById('blackoutAlert');
    const progressSlider = document.getElementById('progressSlider');
    const progressText = document.getElementById('progressText');
    
    // Telemetry Elements
    const speedVal = document.getElementById('speedVal');
    const headingVal = document.getElementById('headingVal');
    const headingCompass = document.getElementById('headingCompass');
    const blackoutDistVal = document.getElementById('blackoutDistVal');
    const uncertaintyVal = document.getElementById('uncertaintyVal');
    const driftPercentVal = document.getElementById('driftPercentVal');
    const absDriftVal = document.getElementById('absDriftVal');
    const driftProgressBar = document.getElementById('driftProgressBar');
    const sihBadge = document.getElementById('sihBadge');
    const mobileStreamBtn = document.getElementById('mobileStreamBtn');

    // State
    let isPlaying = false;
    let playbackInterval = null;
    let speedMultiplier = 1;
    let blackoutForced = false;
    let totalFrames = 1;

    // Compass direction helper
    function getCompassDirection(deg) {
        const directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
        const index = Math.round(deg / 45) % 8;
        return directions[index];
    }

    // 1. Fetch available trips
    async function fetchTrips() {
        try {
            const res = await fetch('/api/trips');
            const data = await res.json();
            tripSelect.innerHTML = '';
            if (data.trips && data.trips.length > 0) {
                data.trips.forEach(t => {
                    const opt = document.createElement('option');
                    opt.value = t;
                    opt.textContent = t;
                    tripSelect.appendChild(opt);
                });
                // Auto-load first trip
                loadTrip(data.trips[0]);
            } else {
                tripSelect.innerHTML = '<option value="">No trips found</option>';
            }
        } catch (err) {
            console.error("Failed to load trips:", err);
        }
    }

    // 2. Load chosen trip into server session
    async function loadTrip(tripId) {
        try {
            pausePlayback();
            const res = await fetch('/api/session/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ trip_id: tripId })
            });
            const data = await res.json();
            if (data.status === 'success') {
                totalFrames = data.session.num_frames;
                navMap.clearPaths();
                navMap.setCenter(data.session.origin.lat, data.session.origin.lon, 17);
                updateUIState(false);
                progressSlider.value = 0;
                progressText.textContent = "0%";
                // Execute initial step to place marker
                stepSimulation();
            }
        } catch (err) {
            console.error("Failed to start session:", err);
        }
    }

    // 3. Step simulation frame
    async function stepSimulation() {
        try {
            const stepFrames = 10 * speedMultiplier; // 100Hz data, 10 frames = 0.1s
            const res = await fetch('/api/engine/step', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ step_size: stepFrames })
            });
            const state = await res.json();
            if (state.finished) {
                pausePlayback();
                return;
            }

            // Update Map
            navMap.updateState(state);

            // Update IMU Chart
            if (state.imu) {
                imuVis.update(state.imu.accel, state.imu.gyro);
            }

            // Update Telemetry Display
            speedVal.textContent = state.speed_kmh.toFixed(1);
            headingVal.textContent = state.heading_deg.toFixed(1) + '°';
            headingCompass.textContent = getCompassDirection(state.heading_deg);
            blackoutDistVal.textContent = state.blackout_distance_m.toFixed(1) + ' m';
            uncertaintyVal.textContent = '±' + state.uncertainty_sigma_m.toFixed(2) + ' m';
            absDriftVal.textContent = state.horizontal_drift_m.toFixed(2) + ' m';
            driftPercentVal.textContent = state.drift_percent.toFixed(1) + '%';
            
            // SIH Benchmark Bar
            const barWidth = Math.min(Math.max((state.drift_percent / 20.0) * 100, 2), 100);
            driftProgressBar.style.width = barWidth + '%';
            if (state.drift_percent > 10.0 && state.blackout_active) {
                driftProgressBar.className = 'bg-red-500 h-2 rounded-full transition-all duration-300';
                sihBadge.className = 'text-[10px] px-1.5 py-0.5 rounded bg-red-950 text-red-400 border border-red-800';
                sihBadge.textContent = 'FAIL (>10%)';
            } else {
                driftProgressBar.className = 'bg-emerald-500 h-2 rounded-full transition-all duration-300';
                sihBadge.className = 'text-[10px] px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800';
                sihBadge.textContent = 'PASS (<10%)';
            }

            // Slider & Progress
            progressSlider.value = state.progress_percent;
            progressText.textContent = state.progress_percent.toFixed(0) + '%';

        } catch (err) {
            console.error("Step error:", err);
            pausePlayback();
        }
    }

    // Playback loop
    function startPlayback() {
        if (isPlaying) return;
        isPlaying = true;
        playBtn.innerHTML = '<span>⏸</span> Pause';
        playBtn.className = 'bg-amber-600 hover:bg-amber-700 text-white px-4 py-1.5 rounded text-xs font-bold transition flex items-center gap-1';
        playbackInterval = setInterval(stepSimulation, 100);
    }

    function pausePlayback() {
        isPlaying = false;
        if (playbackInterval) {
            clearInterval(playbackInterval);
            playbackInterval = null;
        }
        playBtn.innerHTML = '<span>▶</span> Run Playback';
        playBtn.className = 'bg-emerald-600 hover:bg-emerald-700 text-white px-4 py-1.5 rounded text-xs font-bold transition flex items-center gap-1';
    }

    // Blackout UI updater
    function updateUIState(isBlackout) {
        blackoutForced = isBlackout;
        if (isBlackout) {
            statusBadge.className = 'flex items-center space-x-2 px-3 py-1 rounded-full text-xs font-semibold bg-red-950 text-red-400 border border-red-800';
            statusText.textContent = 'DEAD RECKONING';
            blackoutAlert.classList.remove('hidden');
            blackoutBtn.textContent = 'RESTORE GNSS';
            blackoutBtn.className = 'px-3 py-1 rounded text-xs font-bold transition-all duration-200 bg-emerald-600 hover:bg-emerald-700 text-white shadow';
        } else {
            statusBadge.className = 'flex items-center space-x-2 px-3 py-1 rounded-full text-xs font-semibold bg-green-950 text-green-400 border border-green-800';
            statusText.textContent = 'GNSS ACTIVE';
            blackoutAlert.classList.add('hidden');
            blackoutBtn.textContent = 'DROP GNSS';
            blackoutBtn.className = 'px-3 py-1 rounded text-xs font-bold transition-all duration-200 bg-red-600 hover:bg-red-700 text-white shadow';
        }
    }

    // Event Listeners
    loadTripBtn.addEventListener('click', () => {
        if (tripSelect.value) loadTrip(tripSelect.value);
    });

    playBtn.addEventListener('click', () => {
        if (isPlaying) pausePlayback();
        else startPlayback();
    });

    resetBtn.addEventListener('click', async () => {
        pausePlayback();
        await fetch('/api/engine/reset', { method: 'POST' });
        navMap.clearPaths();
        progressSlider.value = 0;
        progressText.textContent = '0%';
        updateUIState(false);
        stepSimulation();
    });

    blackoutBtn.addEventListener('click', async () => {
        const nextState = !blackoutForced;
        const res = await fetch('/api/engine/toggle_blackout', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ forced: nextState })
        });
        const data = await res.json();
        updateUIState(data.blackout_active);
    });

    // Speed buttons
    document.querySelectorAll('.speed-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.speed-btn').forEach(b => {
                b.className = 'speed-btn px-2 py-1 bg-gray-800 rounded border border-gray-700 hover:text-white';
            });
            btn.className = 'speed-btn px-2 py-1 bg-gray-800 rounded border border-gray-700 text-blue-400 font-bold';
            speedMultiplier = parseInt(btn.getAttribute('data-speed'));
        });
    });

    // Mobile Phone IMU Streamer
    const mobileStreamer = new MobileSensorStreamer((imu) => {
        imuVis.update(imu.accel, imu.gyro);
    });

    mobileStreamBtn.addEventListener('click', async () => {
        if (!mobileStreamer.active) {
            const started = await mobileStreamer.start();
            if (started) {
                mobileStreamBtn.textContent = 'Stop Phone IMU';
                mobileStreamBtn.className = 'w-full py-2 bg-red-600 hover:bg-red-700 text-white rounded text-xs font-bold transition';
            }
        } else {
            mobileStreamer.stop();
            mobileStreamBtn.textContent = 'Activate Phone IMU';
            mobileStreamBtn.className = 'w-full py-2 bg-purple-600 hover:bg-purple-700 text-white rounded text-xs font-bold transition';
        }
    });

    // Initialize
    fetchTrips();
});
