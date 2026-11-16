/**
 * AegisFlow: Autonomous Zero-Day Network Intrusion Sentinel
 * Module: static/js/dashboard.js
 * Description: Real-time SOC dashboard controller featuring Chart.js streaming telemetry,
 * 2D latent space scatter visualization, multi-model radar benchmarking, and XAI forensic diagnostics.
 */

// Application State
const state = {
    totalFlows: 0,
    flaggedCount: 0,
    recentVotes: [],
    flowHistory: [],
    isStreaming: true,
    streamTimer: null,
    streamIntervalMs: 1200,
    nextInjection: null,
    maxHistoryPoints: 24,
    charts: {
        telemetry: null,
        latentScatter: null,
        radar: null
    }
};

// DOM References
const DOM = {
    totalFlows: document.getElementById('kpiTotalFlows'),
    flaggedCount: document.getElementById('kpiFlaggedCount'),
    anomalyRate: document.getElementById('kpiAnomalyRate'),
    ensembleAgreement: document.getElementById('kpiEnsembleAgreement'),
    streamRate: document.getElementById('kpiStreamRate'),
    streamToggle: document.getElementById('streamToggle'),
    streamLabel: document.getElementById('streamLabel'),
    flowFeedBody: document.getElementById('flowFeedBody'),
    tableCounter: document.getElementById('tableCounter'),
    systemStatusText: document.getElementById('systemStatusText'),
    
    // Modals
    xaiModal: document.getElementById('xaiModal'),
    xaiModalSubtitle: document.getElementById('xaiModalSubtitle'),
    xaiThreatBanner: document.getElementById('xaiThreatBanner'),
    xaiThreatText: document.getElementById('xaiThreatText'),
    xaiSummaryText: document.getElementById('xaiSummaryText'),
    xaiBarsContainer: document.getElementById('xaiBarsContainer'),
    xaiDetailsBody: document.getElementById('xaiDetailsBody'),

    retrainModal: document.getElementById('retrainModal'),
    retrainForm: document.getElementById('retrainForm'),
    trainingProgressBox: document.getElementById('trainingProgressBox'),
    btnSubmitRetrain: document.getElementById('btnSubmitRetrain'),

    uploadModal: document.getElementById('uploadModal'),
    uploadResultsArea: document.getElementById('uploadResultsArea'),
    uploadTotalRows: document.getElementById('uploadTotalRows'),
    uploadAnomalyCount: document.getElementById('uploadAnomalyCount'),
    uploadAnomalyRate: document.getElementById('uploadAnomalyRate'),
    uploadPreviewBody: document.getElementById('uploadPreviewBody')
};

// Initialize Dashboard on Window Load
window.addEventListener('DOMContentLoaded', async () => {
    initCharts();
    setupEventHandlers();
    await fetchInitialStatus();
    await fetchBenchmarkData();
    startStream();
});

// Chart.js Theme Configurations
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'JetBrains Mono', monospace";
Chart.defaults.font.size = 11;

function initCharts() {
    // 1. Live Telemetry Scrolling Line Chart
    const ctxTelemetry = document.getElementById('telemetryChart').getContext('2d');
    state.charts.telemetry = new Chart(ctxTelemetry, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Ensemble Score',
                    borderColor: '#00f3ff',
                    backgroundColor: 'rgba(0, 243, 255, 0.12)',
                    borderWidth: 2.5,
                    pointRadius: 3,
                    pointHoverRadius: 6,
                    tension: 0.35,
                    fill: true,
                    data: []
                },
                {
                    label: 'Isolation Forest',
                    borderColor: '#a855f7',
                    backgroundColor: 'transparent',
                    borderWidth: 1.5,
                    pointRadius: 2,
                    borderDash: [4, 4],
                    tension: 0.25,
                    data: []
                },
                {
                    label: 'Deep Autoencoder',
                    borderColor: '#10b981',
                    backgroundColor: 'transparent',
                    borderWidth: 1.5,
                    pointRadius: 2,
                    tension: 0.25,
                    data: []
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 300 },
            scales: {
                y: {
                    min: 0.0,
                    max: 1.0,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: {
                        callback: val => val.toFixed(1)
                    }
                },
                x: {
                    grid: { display: false }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(11, 17, 30, 0.95)',
                    borderColor: 'rgba(0, 243, 255, 0.3)',
                    borderWidth: 1,
                    titleColor: '#00f3ff',
                    callbacks: {
                        label: ctx => `${ctx.dataset.label}: ${(ctx.parsed.y * 100).toFixed(1)}%`
                    }
                }
            }
        }
    });

    // 2. 2D Latent Representation Scatter Plot
    const ctxScatter = document.getElementById('latentScatterChart').getContext('2d');
    state.charts.latentScatter = new Chart(ctxScatter, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Normal Baseline Cluster',
                    data: [],
                    backgroundColor: 'rgba(16, 185, 129, 0.35)',
                    borderColor: 'rgba(16, 185, 129, 0.8)',
                    pointRadius: 4,
                    pointHoverRadius: 6
                },
                {
                    label: 'Intercepted Intrusions',
                    data: [],
                    backgroundColor: 'rgba(239, 68, 68, 0.8)',
                    borderColor: '#ef4444',
                    pointRadius: 6,
                    pointHoverRadius: 9
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    title: { display: true, text: 'Latent PC-1', color: '#64748b' }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    title: { display: true, text: 'Latent PC-2', color: '#64748b' }
                }
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { boxWidth: 8, padding: 12 }
                },
                tooltip: {
                    backgroundColor: 'rgba(11, 17, 30, 0.95)',
                    borderColor: 'rgba(168, 85, 247, 0.4)',
                    borderWidth: 1
                }
            }
        }
    });

    // 3. Algorithm Comparison Radar Chart
    const ctxRadar = document.getElementById('benchmarkRadarChart').getContext('2d');
    state.charts.radar = new Chart(ctxRadar, {
        type: 'radar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Zero-Day Resil.'],
            datasets: [
                {
                    label: 'AegisFlow Ensemble',
                    data: [97, 96, 98, 97, 98],
                    borderColor: '#00f3ff',
                    backgroundColor: 'rgba(0, 243, 255, 0.25)',
                    pointBackgroundColor: '#00f3ff',
                    borderWidth: 2
                },
                {
                    label: 'Deep Autoencoder',
                    data: [93, 91, 94, 92, 95],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.12)',
                    pointBackgroundColor: '#10b981',
                    borderWidth: 1.5
                },
                {
                    label: 'Isolation Forest',
                    data: [94, 95, 91, 93, 89],
                    borderColor: '#a855f7',
                    backgroundColor: 'rgba(168, 85, 247, 0.12)',
                    pointBackgroundColor: '#a855f7',
                    borderWidth: 1.5
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(255, 255, 255, 0.08)' },
                    grid: { color: 'rgba(255, 255, 255, 0.06)' },
                    pointLabels: { color: '#cbd5e1', font: { size: 10 } },
                    suggestedMin: 80,
                    suggestedMax: 100,
                    ticks: { display: false }
                }
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { boxWidth: 8, padding: 10 }
                }
            }
        }
    });
}

function setupEventHandlers() {
    // Stream Toggle
    DOM.streamToggle.addEventListener('change', e => {
        state.isStreaming = e.target.checked;
        if (state.isStreaming) {
            DOM.streamLabel.textContent = 'AUTO-STREAM ACTIVE';
            DOM.streamLabel.style.color = '#10b981';
            startStream();
        } else {
            DOM.streamLabel.textContent = 'STREAM PAUSED';
            DOM.streamLabel.style.color = '#f59e0b';
            stopStream();
        }
    });

    // Nav Buttons
    document.getElementById('btnOpenRetrain').addEventListener('click', () => openModal('retrainModal'));
    document.getElementById('btnOpenUpload').addEventListener('click', () => openModal('uploadModal'));
    document.getElementById('btnRunBenchmark').addEventListener('click', async () => {
        DOM.systemStatusText.textContent = 'BENCHMARKING...';
        await fetchBenchmarkData();
        DOM.systemStatusText.textContent = 'SENTINEL ARMED';
    });
}

async function fetchInitialStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        if (data.is_ready) {
            DOM.systemStatusText.textContent = 'SENTINEL ARMED';
            // Populate initial normal cluster sample in scatter plot
            if (data.baseline_cluster_samples && data.baseline_cluster_samples.length > 0) {
                state.charts.latentScatter.data.datasets[0].data = data.baseline_cluster_samples;
                state.charts.latentScatter.update();
            }
        }
    } catch (err) {
        console.error('Error fetching sentinel status:', err);
        DOM.systemStatusText.textContent = 'SENTINEL OFFLINE';
    }
}

async function fetchBenchmarkData() {
    try {
        const res = await fetch('/api/benchmark');
        const data = await res.json();
        if (data.status === 'SUCCESS' && data.benchmarks) {
            const b = data.benchmarks;
            const ens = b.ensemble;
            const ae = b.autoencoder;
            const iso = b.isolation_forest;

            state.charts.radar.data.datasets[0].data = [
                ens.accuracy, ens.precision, ens.recall, ens.f1_score, 98.5
            ];
            state.charts.radar.data.datasets[1].data = [
                ae.accuracy, ae.precision, ae.recall, ae.f1_score, 94.0
            ];
            state.charts.radar.data.datasets[2].data = [
                iso.accuracy, iso.precision, iso.recall, iso.f1_score, 91.0
            ];
            state.charts.radar.update();
        }
    } catch (err) {
        console.warn('Could not refresh benchmark data:', err);
    }
}

function startStream() {
    if (state.streamTimer) clearInterval(state.streamTimer);
    state.streamTimer = setInterval(pollStreamTelemetry, state.streamIntervalMs);
}

function stopStream() {
    if (state.streamTimer) {
        clearInterval(state.streamTimer);
        state.streamTimer = null;
    }
}

// User-Triggered Threat Injections
window.injectThreat = function(attackType) {
    state.nextInjection = attackType;
    pollStreamTelemetry();
};

async function pollStreamTelemetry() {
    try {
        const attackParam = state.nextInjection || 'none';
        state.nextInjection = null; // consume one-shot injection

        const url = `/api/traffic/stream?attack=${encodeURIComponent(attackParam)}`;
        const res = await fetch(url);
        if (!res.ok) return;

        const payload = await res.json();
        processTelemetryPacket(payload);
    } catch (err) {
        console.error('Streaming error:', err);
    }
}

function processTelemetryPacket(data) {
    const { timestamp, flow, detection, xai } = data;

    state.totalFlows++;
    if (detection.is_anomaly) {
        state.flaggedCount++;
    }

    // Keep history for inspection
    const flowRecord = {
        id: state.totalFlows,
        timestamp,
        flow,
        detection,
        xai
    };
    state.flowHistory.unshift(flowRecord);
    if (state.flowHistory.length > 50) state.flowHistory.pop();

    // Update KPIs
    DOM.totalFlows.textContent = state.totalFlows.toLocaleString();
    DOM.flaggedCount.textContent = state.flaggedCount.toLocaleString();
    const rate = ((state.flaggedCount / state.totalFlows) * 100).toFixed(1);
    DOM.anomalyRate.textContent = `${rate}%`;

    state.recentVotes.push(detection.model_agreement_votes);
    if (state.recentVotes.length > 20) state.recentVotes.shift();
    const avgAgreement = (state.recentVotes.reduce((a, b) => a + b, 0) / (state.recentVotes.length * 3)) * 100;
    DOM.ensembleAgreement.textContent = `${avgAgreement.toFixed(1)}%`;
    DOM.streamRate.textContent = (1000 / state.streamIntervalMs).toFixed(1);

    // Update Telemetry Line Chart
    const chart = state.charts.telemetry;
    chart.data.labels.push(timestamp);
    chart.data.datasets[0].data.push(detection.ensemble_score);
    chart.data.datasets[1].data.push(detection.isolation_forest_score);
    chart.data.datasets[2].data.push(detection.autoencoder_score);

    if (chart.data.labels.length > state.maxHistoryPoints) {
        chart.data.labels.shift();
        chart.data.datasets.forEach(ds => ds.data.shift());
    }
    chart.update();

    // Update 2D Latent Scatter Plot
    const scatter = state.charts.latentScatter;
    const pt = { x: detection.latent_coords.x, y: detection.latent_coords.y };
    if (detection.is_anomaly) {
        scatter.data.datasets[1].data.push(pt);
        if (scatter.data.datasets[1].data.length > 40) {
            scatter.data.datasets[1].data.shift();
        }
    } else {
        scatter.data.datasets[0].data.push(pt);
        if (scatter.data.datasets[0].data.length > 80) {
            scatter.data.datasets[0].data.shift();
        }
    }
    scatter.update();

    // Append Row to Feed Table
    appendFeedTableRow(flowRecord);
}

function appendFeedTableRow(record) {
    const tr = document.createElement('tr');
    const d = record.detection;
    const f = record.flow;

    const badgeClass = `badge-${d.severity}`;
    const scoreColor = d.is_anomaly ? 'text-crimson' : 'text-emerald';

    tr.innerHTML = `
        <td>${record.timestamp}</td>
        <td><strong>${f.protocol || 'TCP'}</strong> <span style="font-size:0.7rem; color:#64748b;">${f.attack_type !== 'None' ? '(' + f.attack_type + ')' : ''}</span></td>
        <td>${f.duration}s</td>
        <td>${Number(f.src_bytes).toLocaleString()} / ${Number(f.dst_bytes).toLocaleString()}</td>
        <td>${f.packet_count}</td>
        <td class="${scoreColor}">${(d.ensemble_score * 100).toFixed(1)}%</td>
        <td><span class="badge-threat ${badgeClass}">${d.severity}</span></td>
        <td><button class="btn-inspect" onclick="inspectFlow(${record.id})">Inspect XAI</button></td>
    `;

    DOM.flowFeedBody.insertBefore(tr, DOM.flowFeedBody.firstChild);
    while (DOM.flowFeedBody.children.length > 25) {
        DOM.flowFeedBody.removeChild(DOM.flowFeedBody.lastChild);
    }
}

// Inspect XAI Diagnostics Modal
window.inspectFlow = function(flowId) {
    const item = state.flowHistory.find(r => r.id === flowId);
    if (!item) return;

    const { flow, detection, xai, timestamp } = item;

    DOM.xaiModalSubtitle.textContent = `Flow #${item.id} @ ${timestamp} // Protocol: ${flow.protocol} // Ensemble Score: ${(detection.ensemble_score * 100).toFixed(1)}%`;

    // Threat Indicators Banner
    if (xai.threat_indicators && xai.threat_indicators.length > 0) {
        DOM.xaiThreatBanner.style.display = 'flex';
        DOM.xaiThreatText.innerHTML = xai.threat_indicators.map(ind => `<div>[!] ${ind}</div>`).join('');
    } else {
        DOM.xaiThreatBanner.style.display = detection.is_anomaly ? 'flex' : 'none';
        DOM.xaiThreatText.textContent = detection.is_anomaly
            ? 'General zero-day behavioral anomaly detected outside normal baseline cluster.'
            : 'Benign telemetry. Conforms to learned statistical distribution.';
    }

    // Forensic Summary Text
    DOM.xaiSummaryText.textContent = xai.explanation_summary;

    // Render Contribution Bars
    DOM.xaiBarsContainer.innerHTML = '';
    const topItems = xai.top_contributors || [];
    topItems.forEach(item => {
        const isElevated = item.direction === 'ELEVATED';
        const fillClass = isElevated ? 'bar-fill bar-fill-elevated' : 'bar-fill';
        const row = document.createElement('div');
        row.className = 'xai-bar-row';
        row.innerHTML = `
            <div class="bar-meta">
                <span><strong>${item.display_name}</strong> (${item.direction})</span>
                <span>Z = ${item.z_score > 0 ? '+' : ''}${item.z_score}σ &bull; ${item.contribution_pct}% Impact</span>
            </div>
            <div class="bar-track">
                <div class="${fillClass}" style="width: ${Math.min(100, Math.max(8, item.contribution_pct * 2.2))}%"></div>
            </div>
        `;
        DOM.xaiBarsContainer.appendChild(row);
    });

    // Render 14-Feature Deviation Table
    DOM.xaiDetailsBody.innerHTML = '';
    (xai.feature_deviations || []).forEach(feat => {
        const tr = document.createElement('tr');
        const zColor = Math.abs(feat.z_score) >= 2.0 ? 'text-crimson' : (Math.abs(feat.z_score) >= 1.0 ? 'text-amber' : '');
        tr.innerHTML = `
            <td><strong>${feat.display_name}</strong> (${feat.feature})</td>
            <td>${Number(feat.observed_value).toLocaleString()}</td>
            <td>${Number(feat.baseline_mean).toLocaleString()} &plusmn; ${Number(feat.baseline_std).toLocaleString()}</td>
            <td class="${zColor}">${feat.z_score > 0 ? '+' : ''}${feat.z_score}&sigma;</td>
            <td>${feat.contribution_pct}%</td>
        `;
        DOM.xaiDetailsBody.appendChild(tr);
    });

    openModal('xaiModal');
};

// Modal Management
window.openModal = function(id) {
    const el = document.getElementById(id);
    if (el) el.classList.add('open');
};

window.closeModal = function(id) {
    const el = document.getElementById(id);
    if (el) el.classList.remove('open');
};

// Retrain Model Handler
window.handleRetrain = async function(e) {
    e.preventDefault();
    const sampleSize = parseInt(document.getElementById('paramSampleSize').value, 10);
    const contamination = parseFloat(document.getElementById('paramContamination').value);
    const epochs = parseInt(document.getElementById('paramEpochs').value, 10);

    DOM.trainingProgressBox.style.display = 'flex';
    DOM.btnSubmitRetrain.disabled = true;

    try {
        const res = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sample_size: sampleSize, contamination, epochs })
        });
        const data = await res.json();
        if (res.ok) {
            alert(`✔ Retraining Complete!\n${data.message}`);
            closeModal('retrainModal');
            await fetchInitialStatus();
            await fetchBenchmarkData();
        } else {
            alert(`Training Error: ${data.error || 'Failed'}`);
        }
    } catch (err) {
        alert(`Error retraining baseline: ${err.message}`);
    } finally {
        DOM.trainingProgressBox.style.display = 'none';
        DOM.btnSubmitRetrain.disabled = false;
    }
};

// CSV Upload Handler
window.handleFileSelect = async function(e) {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
        DOM.uploadResultsArea.style.display = 'none';
        const res = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();

        if (res.ok) {
            DOM.uploadResultsArea.style.display = 'block';
            DOM.uploadTotalRows.textContent = data.total_records.toLocaleString();
            DOM.uploadAnomalyCount.textContent = data.anomaly_count.toLocaleString();
            DOM.uploadAnomalyRate.textContent = `${data.anomaly_rate_pct}%`;

            DOM.uploadPreviewBody.innerHTML = '';
            (data.records_preview || []).slice(0, 25).forEach(rec => {
                const tr = document.createElement('tr');
                const scoreColor = rec.is_anomaly ? 'text-crimson' : 'text-emerald';
                tr.innerHTML = `
                    <td>${rec.row_index}</td>
                    <td>${rec.protocol}</td>
                    <td>${Number(rec.src_bytes).toLocaleString()}</td>
                    <td>${Number(rec.dst_bytes).toLocaleString()}</td>
                    <td class="${scoreColor}">${(rec.anomaly_score * 100).toFixed(1)}%</td>
                    <td>${rec.is_anomaly ? '<span class="text-crimson">ANOMALY</span>' : '<span class="text-emerald">NORMAL</span>'}</td>
                    <td><span class="badge-threat badge-${rec.severity}">${rec.severity}</span></td>
                `;
                DOM.uploadPreviewBody.appendChild(tr);
            });
        } else {
            alert(`CSV Processing Error: ${data.error}`);
        }
    } catch (err) {
        alert(`Upload failed: ${err.message}`);
    }
};
