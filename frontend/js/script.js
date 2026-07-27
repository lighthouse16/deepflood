// DeepFlood Hydrological Intelligence Logic
let globalData = [];
let mainChartInstance = null;

document.addEventListener('DOMContentLoaded', async () => {
    initSPA();
    await loadComparisonData();
    setupEventListeners();
});

// 1. Single Page Application (SPA) Navigation
function initSPA() {
    const navItems = document.querySelectorAll('.nav-item');
    const views = document.querySelectorAll('.view-section');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = item.getAttribute('data-target');

            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');

            views.forEach(view => {
                view.classList.remove('active');
                if (view.id === targetId) {
                    view.classList.add('active');
                }
            });

            if (targetId === 'view-dashboard' && mainChartInstance) {
                mainChartInstance.resize();
            }
        });
    });
}

// 2. Load CSV
async function loadComparisonData() {
    try {
        // Add cache buster to ensure live GitHub Pages data is fetched immediately without browser caching
        const response = await fetch('data/model_comparison_data.csv?v=' + Date.now(), { cache: 'no-store' });
        if (!response.ok) throw new Error('Data not found');
        const csvText = await response.text();
        
        globalData = parseCSV(csvText);
        if (globalData.length > 0) {
            updateDashboardKPIs(globalData);
            renderMainChart(globalData);
            populateDataTable(globalData);
            initWhatIfSimulator(globalData);
        }
    } catch (error) {
        console.error('Data Load Error:', error);
    }
}

function parseCSV(text) {
    const lines = text.trim().split('\n');
    if (lines.length < 2) return [];

    const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
    const dateIdx = headers.indexOf('date');
    const obsIdx = headers.indexOf('observed_streamflow');
    const prcpIdx = headers.indexOf('prcp_mm');
    const predIdx = headers.indexOf('predicted_streamflow');
    const errIdx = headers.indexOf('error_abs');

    if (dateIdx === -1 || predIdx === -1) return [];

    const data = [];
    for (let i = 1; i < lines.length; i++) {
        const cols = lines[i].split(',').map(c => c.trim());
        if (cols.length <= Math.max(dateIdx, predIdx)) continue;

        data.push({
            date: cols[dateIdx],
            observed: obsIdx !== -1 ? parseFloat(cols[obsIdx]) || 0 : 0,
            prcp: prcpIdx !== -1 ? parseFloat(cols[prcpIdx]) || 0 : 0,
            predicted: parseFloat(cols[predIdx]) || 0,
            error: errIdx !== -1 ? parseFloat(cols[errIdx]) || 0 : 0
        });
    }
    return data;
}

// 3. Update KPIs
function updateDashboardKPIs(data) {
    let peakPred = data[0];
    let peakObs = data[0];

    for (const row of data) {
        if (row.predicted > peakPred.predicted) peakPred = row;
        if (row.observed > peakObs.observed) peakObs = row;
    }

    document.getElementById('kpi-peak-pred').textContent = `${peakPred.predicted.toFixed(1)} m3/s`;
    document.getElementById('kpi-peak-date').textContent = `Peak on ${peakPred.date}`;
    document.getElementById('kpi-peak-obs').textContent = `${peakObs.observed.toFixed(1)} m3/s`;

    const riskLevel = document.getElementById('kpi-risk-level');
    const riskDesc = document.getElementById('kpi-risk-desc');

    riskLevel.className = 'kpi-value';

    if (peakPred.predicted < 500) {
        riskLevel.textContent = 'SAFE';
        riskLevel.classList.add('status-safe');
        riskDesc.textContent = 'Normal flow';
    } else if (peakPred.predicted < 1500) {
        riskLevel.textContent = 'MODERATE';
        riskLevel.classList.add('status-warn');
        riskDesc.textContent = 'Monitoring status';
    } else {
        riskLevel.textContent = 'CRITICAL';
        riskLevel.classList.add('status-danger');
        riskDesc.textContent = 'Flood surge risk';
    }
}

// 4. Render Chart.js Hydrograph & Scatter Plot
let scatterChartInstance = null;

function renderMainChart(dataSubset) {
    const ctx = document.getElementById('mainChart').getContext('2d');
    
    if (mainChartInstance) {
        mainChartInstance.destroy();
    }

    const labels = dataSubset.map(d => d.date);
    const obsValues = dataSubset.map(d => d.observed);
    const predValues = dataSubset.map(d => d.predicted);
    const prcpValues = dataSubset.map(d => d.prcp);

    mainChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Predicted Flow',
                    data: predValues,
                    borderColor: '#2563eb', // accent-blue
                    backgroundColor: 'rgba(37, 99, 235, 0.05)',
                    borderWidth: 2,
                    pointRadius: dataSubset.length < 40 ? 3 : 0,
                    pointHoverRadius: 5,
                    fill: true,
                    tension: 0.1,
                    yAxisID: 'y'
                },
                {
                    label: 'Observed Flow',
                    data: obsValues,
                    borderColor: '#111827', // text-primary
                    borderWidth: 1.5,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    fill: false,
                    tension: 0.1,
                    yAxisID: 'y'
                },
                {
                    label: 'Rainfall',
                    data: prcpValues,
                    type: 'bar',
                    backgroundColor: '#cbd5e1',
                    borderColor: '#94a3b8',
                    borderWidth: 1,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#4b5563',
                        font: { family: 'Inter', size: 12 },
                        usePointStyle: true,
                        boxWidth: 8
                    }
                },
                tooltip: {
                    backgroundColor: '#111827',
                    titleFont: { family: 'Inter', size: 12, weight: '600' },
                    bodyFont: { family: 'Inter', size: 12 },
                    padding: 12,
                    cornerRadius: 4,
                    displayColors: false
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#6b7280', font: { family: 'Inter', size: 11 }, maxTicksLimit: 10 }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    grid: { color: '#f3f4f6' },
                    ticks: { color: '#6b7280', font: { family: 'Inter', size: 11 } },
                    title: { display: true, text: 'Streamflow (m3/s)', color: '#6b7280', font: { size: 11 } }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    reverse: true, // Inverted rainfall axis (standard hydrograph display)
                    grid: { display: false },
                    ticks: { color: '#6b7280', font: { family: 'Inter', size: 11 } },
                    title: { display: true, text: 'Precipitation (mm)', color: '#6b7280', font: { size: 11 } },
                    min: 0,
                    max: Math.max(...prcpValues, 50) * 4 // Push bars to top 25%
                }
            }
        }
    });

    renderScatterChart(dataSubset);
}

function renderScatterChart(dataSubset) {
    const canvas = document.getElementById('scatterChart');
    if (!canvas) return; // Only exists in metrics view
    const ctx = canvas.getContext('2d');
    
    if (scatterChartInstance) {
        scatterChartInstance.destroy();
    }

    const scatterData = dataSubset.map(d => ({ x: d.observed, y: d.predicted }));
    const maxVal = Math.max(...dataSubset.map(d => Math.max(d.observed, d.predicted)));
    
    // Ideal y=x line
    const idealLine = [{x: 0, y: 0}, {x: maxVal, y: maxVal}];

    scatterChartInstance = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Predicted vs Observed',
                    data: scatterData,
                    backgroundColor: 'rgba(37, 99, 235, 0.5)',
                    borderColor: '#2563eb',
                    borderWidth: 1,
                    pointRadius: 4,
                    pointHoverRadius: 6
                },
                {
                    label: 'Perfect Alignment (y=x)',
                    data: idealLine,
                    type: 'line',
                    borderColor: '#9ca3af',
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#4b5563', font: { family: 'Inter', size: 12 }, usePointStyle: true }
                },
                tooltip: {
                    backgroundColor: '#111827',
                    callbacks: {
                        label: function(context) {
                            if (context.datasetIndex === 1) return 'Ideal alignment';
                            return `Obs: ${context.parsed.x.toFixed(1)}, Pred: ${context.parsed.y.toFixed(1)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: { display: true, text: 'Observed Flow (m3/s)', color: '#6b7280' },
                    grid: { color: '#f3f4f6' },
                    min: 0
                },
                y: {
                    type: 'linear',
                    position: 'left',
                    title: { display: true, text: 'Predicted Flow (m3/s)', color: '#6b7280' },
                    grid: { color: '#f3f4f6' },
                    min: 0
                }
            }
        }
    });
}

// 5. Data Table
function populateDataTable(dataSubset) {
    const tbody = document.getElementById('data-table-body');
    tbody.innerHTML = '';

    dataSubset.forEach(row => {
        const tr = document.createElement('tr');
        
        tr.innerHTML = `
            <td>${row.date}</td>
            <td class="num-col">${row.prcp.toFixed(1)}</td>
            <td class="num-col">${row.observed.toFixed(1)}</td>
            <td class="num-col" style="font-weight: 500; color: #111827;">${row.predicted.toFixed(1)}</td>
            <td class="num-col" style="color: #6b7280;">${row.error.toFixed(1)}</td>
        `;
        tbody.appendChild(tr);
    });
}

// 6. Simulator
function initWhatIfSimulator(data) {
    let basePeak = 0;
    data.forEach(d => { if (d.predicted > basePeak) basePeak = d.predicted; });

    const slider = document.getElementById('rain-slider');
    const sliderVal = document.getElementById('slider-val');
    const basePeakEl = document.getElementById('sim-base-peak');
    const adjPeakEl = document.getElementById('sim-adj-peak');
    const riskBadge = document.getElementById('sim-risk-badge');

    basePeakEl.textContent = `${basePeak.toFixed(1)}`;
    
    function updateSim() {
        const extraRain = parseFloat(slider.value) || 0;
        sliderVal.textContent = `+${extraRain} mm`;

        const adjustedPeak = basePeak + (extraRain * 9.2);
        adjPeakEl.textContent = `${adjustedPeak.toFixed(1)}`;

        riskBadge.className = 'sim-val';
        if (adjustedPeak < 600) {
            riskBadge.textContent = 'SAFE';
            riskBadge.classList.add('status-safe');
        } else if (adjustedPeak < 1500) {
            riskBadge.textContent = 'MODERATE';
            riskBadge.classList.add('status-warn');
        } else {
            riskBadge.textContent = 'CRITICAL';
            riskBadge.classList.add('status-danger');
        }
    }

    slider.addEventListener('input', updateSim);
    updateSim();
}

// 7. Event Listeners
function setupEventListeners() {
    const chartBtns = document.querySelectorAll('.chart-filters .filter-btn');
    chartBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            chartBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const range = btn.getAttribute('data-range');
            let filtered = globalData;

            if (range === 'surge') {
                filtered = globalData.filter(d => d.date >= '2024-10-15' && d.date <= '2024-11-15');
            } else if (range === 'dry') {
                filtered = globalData.filter(d => d.date >= '2025-03-01' && d.date <= '2025-06-30');
            }

            renderMainChart(filtered);
        });
    });

    const tblBtns = document.querySelectorAll('.table-filters .filter-btn');
    tblBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tblBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const filter = btn.getAttribute('data-filter');
            let filtered = globalData;

            if (filter === 'flood') {
                filtered = globalData.filter(d => d.predicted >= 1000 || d.observed >= 1000);
            }

            populateDataTable(filtered);
        });
    });
}
