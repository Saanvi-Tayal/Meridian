/**
 * Sensors and IMU Waveform Visualizer
 */

class IMUVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.chart = null;
        this.maxPoints = 40;
        this.labels = Array(this.maxPoints).fill('');
        this.accelData = Array(this.maxPoints).fill(0);
        this.gyroData = Array(this.maxPoints).fill(0);
        this.initChart();
    }

    initChart() {
        if (!this.canvas) return;
        const ctx = this.canvas.getContext('2d');
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.labels,
                datasets: [
                    {
                        label: 'Accel Norm (m/s²)',
                        data: this.accelData,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        tension: 0.3
                    },
                    {
                        label: 'Yaw Rate (°/s)',
                        data: this.gyroData,
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        tension: 0.3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: { display: false },
                    y: {
                        grid: { color: '#1e293b' },
                        ticks: { color: '#64748b', font: { size: 9 } }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        labels: { color: '#94a3b8', font: { size: 10 }, boxWidth: 12 }
                    }
                }
            }
        });
    }

    update(accel, gyro) {
        if (!this.chart) return;
        
        // Accel magnitude without gravity (norm - 9.81)
        const aNorm = Math.sqrt(accel[0]**2 + accel[1]**2 + accel[2]**2);
        // Yaw rate in deg/s
        const gYawDeg = Math.abs(gyro[2]) * (180.0 / Math.PI);

        this.accelData.push(aNorm);
        this.accelData.shift();
        
        this.gyroData.push(gYawDeg);
        this.gyroData.shift();

        this.chart.update('none');
    }
}

/**
 * Mobile DeviceMotionEvent Handler for Live In-Vehicle Streaming
 */
class MobileSensorStreamer {
    constructor(onSensorUpdate) {
        this.active = false;
        this.onSensorUpdate = onSensorUpdate;
    }

    async start() {
        if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
            try {
                const response = await DeviceMotionEvent.requestPermission();
                if (response !== 'granted') {
                    alert('Permission to access device motion was denied.');
                    return false;
                }
            } catch (err) {
                console.error(err);
            }
        }

        window.addEventListener('devicemotion', this.handleMotion.bind(this));
        this.active = true;
        return true;
    }

    stop() {
        window.removeEventListener('devicemotion', this.handleMotion.bind(this));
        this.active = false;
    }

    handleMotion(event) {
        if (!this.active) return;
        const acc = event.accelerationIncludingGravity || { x: 0, y: 0, z: 9.81 };
        const rot = event.rotationRate || { alpha: 0, beta: 0, gamma: 0 };

        // Convert to rad/s
        const gyroRad = [
            (rot.beta || 0) * (Math.PI / 180.0),
            (rot.gamma || 0) * (Math.PI / 180.0),
            (rot.alpha || 0) * (Math.PI / 180.0)
        ];

        if (this.onSensorUpdate) {
            this.onSensorUpdate({
                accel: [acc.x || 0, acc.y || 0, acc.z || 9.81],
                gyro: gyroRad
            });
        }
    }
}
