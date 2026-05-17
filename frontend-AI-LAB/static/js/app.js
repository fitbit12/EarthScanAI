const uploadedFiles = { pre: false, post: false };
const uploadedFileObjects = { pre: null, post: null };
const uploadedImageSrc = { pre: '', post: '' };
let confidenceChart = null;

const damageClassMap = {
  'no-damage': 'no-damage',
  'minor-damage': 'minor',
  'major-damage': 'major',
  destroyed: 'destroyed',
};

function percentText(value) {
  const number = Number(value || 0);
  return `${number.toFixed(2)}%`;
}

function metricText(metric) {
  if (!metric) return 'N/A';
  return `${percentText(metric.mean)} +/- ${Number(metric.std || 0).toFixed(2)}%`;
}

function showResultsEntrance() {
  const entranceCards = document.querySelectorAll('#screen-results .fade-in');
  entranceCards.forEach(card => {
    const entrance = card.dataset.entrance;
    const delay = entrance === 'image-right' || entrance === 'model-right' ? '0.15s' : entrance === 'chart' ? '0.3s' : '0s';
    card.style.animationDelay = delay;
    card.classList.remove('fade-in');
    void card.offsetWidth;
    card.classList.add('fade-in');
  });
}

function showScreen(tabName) {
  document.querySelectorAll('.top-nav-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
  document.getElementById(`screen-${tabName}`).classList.add('active');
  if (tabName === 'results') showResultsEntrance();
}

function syncResultThumb(key, src) {
  const resultThumb = document.getElementById(`result-thumb-${key}`);
  const placeholder = document.getElementById(`result-thumb-${key}-placeholder`);
  if (!resultThumb || !placeholder) return;
  if (src) {
    resultThumb.src = src;
    resultThumb.style.display = 'block';
    placeholder.style.display = 'none';
  } else {
    resultThumb.removeAttribute('src');
    resultThumb.style.display = 'none';
    placeholder.style.display = 'block';
  }
}

function updateAnalyzeButtonState() {
  const btn = document.getElementById('analyze-btn');
  if (uploadedFiles.pre && uploadedFiles.post) {
    btn.disabled = false;
    btn.classList.add('enabled');
  } else {
    btn.disabled = true;
    btn.classList.remove('enabled');
  }
}

function clearUploadSelection(key) {
  uploadedFiles[key] = false;
  uploadedFileObjects[key] = null;
  uploadedImageSrc[key] = '';

  const input = document.getElementById(`input-${key}`);
  const preview = document.getElementById(`preview-${key}`);
  const filename = document.getElementById(`filename-${key}`);
  const badge = document.getElementById(`badge-${key}`);
  const clearBtn = document.getElementById(`clear-${key}`);

  if (input) input.value = '';
  if (preview) {
    preview.removeAttribute('src');
    preview.style.display = 'none';
  }
  if (filename) {
    filename.textContent = '';
    filename.style.display = 'none';
  }
  if (badge) badge.style.display = 'none';
  if (clearBtn) clearBtn.style.display = 'none';

  syncResultThumb(key, '');
  updateAnalyzeButtonState();
}

function resetUploadScreen() {
  clearUploadSelection('pre');
  clearUploadSelection('post');
}

function setupUploadBox(inputId, previewId, filenameId, key) {
  const input = document.getElementById(inputId);
  const preview = document.getElementById(previewId);
  const filename = document.getElementById(filenameId);
  const badge = document.getElementById(`badge-${key}`);
  const clearBtn = document.getElementById(`clear-${key}`);

  if (clearBtn) {
    clearBtn.addEventListener('click', event => {
      event.preventDefault();
      event.stopPropagation();
      clearUploadSelection(key);
    });
  }

  input.addEventListener('change', () => {
    const file = input.files[0];
    if (!file) return;

    uploadedFiles[key] = true;
    uploadedFileObjects[key] = file;
    filename.textContent = file.name;
    filename.style.display = 'block';
    if (badge) badge.style.display = 'inline-flex';
    if (clearBtn) clearBtn.style.display = 'inline-flex';

    if (file.type.startsWith('image/') || /\.(tif|tiff)$/i.test(file.name)) {
      const reader = new FileReader();
      reader.onload = event => {
        preview.src = event.target.result;
        preview.style.display = 'block';
        uploadedImageSrc[key] = event.target.result;
        syncResultThumb(key, event.target.result);
      };
      reader.readAsDataURL(file);
    }

    updateAnalyzeButtonState();
  });
}

function setBadgeState(element, prediction) {
  if (!element || !prediction) return;
  element.textContent = prediction.display_label;
  element.className = `damage-badge ${damageClassMap[prediction.label] || 'major'}`;
}

function updateModelCard(prefix, modelData) {
  setBadgeState(document.getElementById(`${prefix}-prediction-badge`), modelData.prediction);
  modelData.class_scores.forEach((item, index) => {
    const score = document.getElementById(`${prefix}-score-${index}`);
    const fill = document.getElementById(`${prefix}-fill-${index}`);
    if (score) score.textContent = percentText(item.percentage);
    if (fill) fill.style.width = `${Math.max(0, Math.min(100, item.percentage))}%`;
  });

  const accuracy = document.getElementById(`${prefix}-accuracy-value`);
  if (accuracy) {
    accuracy.textContent = metricText(modelData.metrics.five_run_mean_std.accuracy);
  }
}

function buildDatasets(chartPayload, selectedModel) {
  const datasets = chartPayload.datasets.filter(dataset => {
    if (selectedModel === 'cnn') return dataset.label === 'CNN';
    if (selectedModel === 'cnn_knn' || selectedModel === 'cnnknn') return dataset.label === 'CNN+KNN';
    return true;
  });

  return datasets.map(dataset => ({
    label: dataset.label,
    data: dataset.data,
    backgroundColor: dataset.label === 'CNN'
      ? 'rgba(106, 140, 74, 0.7)'
      : 'rgba(212, 118, 42, 0.7)',
    borderColor: dataset.label === 'CNN'
      ? 'rgba(106, 140, 74, 1)'
      : 'rgba(212, 118, 42, 1)',
    borderWidth: 1,
  }));
}

function updateConfidenceChart(chartPayload, selectedModel) {
  const canvas = document.getElementById('confidence-chart');
  if (!canvas || !window.Chart) return;

  const chartData = {
    labels: chartPayload.labels,
    datasets: buildDatasets(chartPayload, selectedModel),
  };

  if (confidenceChart) {
    confidenceChart.data = chartData;
    confidenceChart.update();
    return;
  }

  confidenceChart = new Chart(canvas, {
    type: 'bar',
    data: chartData,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: {
          position: 'top',
          labels: {
            color: '#c8b898',
            usePointStyle: true,
            pointStyle: 'rectRounded',
            boxWidth: 12,
            boxHeight: 12,
          },
        },
        tooltip: {
          enabled: true,
          backgroundColor: 'rgba(18, 13, 8, 0.95)',
          titleColor: '#e8dcc8',
          bodyColor: '#c8b898',
          borderColor: 'rgba(101, 78, 45, 0.3)',
          borderWidth: 1,
        },
      },
      scales: {
        x: {
          ticks: { color: '#8a7a62' },
          grid: { color: 'rgba(101, 78, 45, 0.15)' },
          border: { color: 'rgba(101, 78, 45, 0.15)' },
        },
        y: {
          beginAtZero: true,
          suggestedMax: 100,
          ticks: {
            color: '#8a7a62',
            callback: value => `${value}%`,
          },
          grid: { color: 'rgba(101, 78, 45, 0.15)' },
          border: { color: 'rgba(101, 78, 45, 0.15)' },
        },
      },
    },
  });
}

function updateHeatmap(heatmap) {
  const image = document.getElementById('change-heatmap');
  const method = document.getElementById('heatmap-method');
  const score = document.getElementById('heatmap-change-score');
  if (!heatmap || !image) return;

  ensureHeatmapLegend();
  image.src = heatmap.image_data_url;
  if (method) method.textContent = heatmap.method;
  if (score) {
    score.textContent = `Mean change: ${Number(heatmap.mean_change || 0).toFixed(4)}`;
  }
}

function ensureHeatmapLegend() {
  if (document.querySelector('.heatmap-legend')) return;

  const image = document.getElementById('change-heatmap');
  if (!image || !image.parentElement) return;

  const legend = document.createElement('div');
  legend.className = 'heatmap-legend';
  legend.setAttribute('aria-label', 'Heatmap color legend');
  legend.innerHTML = `
    <div class="heatmap-legend-item">
      <span class="heatmap-swatch heatmap-swatch-high"></span>
      <span>Red / yellow: high visual change, possible damaged area</span>
    </div>
    <div class="heatmap-legend-item">
      <span class="heatmap-swatch heatmap-swatch-medium"></span>
      <span>Green / cyan: moderate visual change</span>
    </div>
    <div class="heatmap-legend-item">
      <span class="heatmap-swatch heatmap-swatch-low"></span>
      <span>Blue / purple: low visual change</span>
    </div>
  `;
  image.insertAdjacentElement('afterend', legend);
}

function updateResultsNote(data) {
  const note = document.getElementById('results-note');
  if (!note) return;

  const cnn = data.models.cnn.prediction;
  const knn = data.models.cnn_knn.prediction;
  note.textContent = [
    `CNN predicted ${cnn.display_label} (${percentText(cnn.confidence_percentage)} confidence).`,
    `CNN+KNN predicted ${knn.display_label} (${percentText(knn.confidence_percentage)} confidence).`,
    data.comparison.message,
    `Heatmap colors: red/yellow = high visual change or possible damaged area, green/cyan = moderate change, blue/purple = low change.`,
    `The heatmap is a visual support map; the final damage class comes from the trained CNN and CNN+KNN models.`,
    `Final evaluation uses 5-run mean +/- standard deviation; backend model uses best run 02.`,
  ].join('\n');
}

function applyPredictionResponse(data) {
  updateModelCard('cnn', data.models.cnn);
  updateModelCard('cnnknn', data.models.cnn_knn);
  updateConfidenceChart(data.charts.confidence, data.selected_model);
  updateHeatmap(data.heatmap);
  updateResultsNote(data);
}

function showError(message) {
  const note = document.getElementById('results-note');
  if (note) note.textContent = message;
  showScreen('results');
}

async function loadMetrics() {
  try {
    const response = await fetch('/api/metrics');
    if (!response.ok) return;
    const data = await response.json();
    const bestAccuracy = document.getElementById('about-best-accuracy');
    const cnnAccuracy = document.getElementById('cnn-accuracy-value');
    const knnAccuracy = document.getElementById('cnnknn-accuracy-value');

    if (bestAccuracy) {
      bestAccuracy.textContent = percentText(data.models.cnn_knn.five_run_mean_std.accuracy.mean);
    }
    if (cnnAccuracy) {
      cnnAccuracy.textContent = metricText(data.models.cnn.five_run_mean_std.accuracy);
    }
    if (knnAccuracy) {
      knnAccuracy.textContent = metricText(data.models.cnn_knn.five_run_mean_std.accuracy);
    }
  } catch (error) {
    console.warn('Could not load metrics:', error);
  }
}

async function analyzeImages() {
  const analyzeBtn = document.getElementById('analyze-btn');
  if (!analyzeBtn.classList.contains('enabled')) return;

  const form = document.getElementById('upload-form');
  const loading = document.getElementById('loading-overlay');
  const msgEl = document.getElementById('loading-msg');
  const modelSelect = document.getElementById('model-select');

  const messages = [
    'Reading satellite imagery...',
    'Preprocessing 256x256 pre/post pair...',
    'Running CNN model...',
    'Running CNN + KNN model...',
    'Preparing confidence chart...',
  ];

  form.style.display = 'none';
  loading.classList.add('visible');

  let step = 0;
  const steps = document.querySelectorAll('.loading-step');
  const msgInterval = setInterval(() => {
    msgEl.classList.add('fade');
    setTimeout(() => {
      step = Math.min(step + 1, messages.length - 1);
      msgEl.textContent = messages[step];
      steps.forEach((item, index) => {
        item.classList.remove('active', 'done');
        if (index < step) item.classList.add('done');
        if (index === step) item.classList.add('active');
      });
      msgEl.classList.remove('fade');
    }, 300);
  }, 1200);

  try {
    const payload = new FormData();
    payload.append('pre_image', uploadedFileObjects.pre);
    payload.append('post_image', uploadedFileObjects.post);
    payload.append('model', modelSelect.value);

    const response = await fetch('/api/predict', {
      method: 'POST',
      body: payload,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'Prediction failed.');
    }

    applyPredictionResponse(data);
    showScreen('results');
  } catch (error) {
    showError(`Prediction failed: ${error.message}`);
  } finally {
    clearInterval(msgInterval);
    loading.classList.remove('visible');
    form.style.display = 'flex';
    msgEl.textContent = messages[0];
    steps.forEach((item, index) => {
      item.classList.remove('active', 'done');
      if (index === 0) item.classList.add('active');
    });
  }
}

function initializeFrontend() {
  syncResultThumb('pre', '');
  syncResultThumb('post', '');
  setupUploadBox('input-pre', 'preview-pre', 'filename-pre', 'pre');
  setupUploadBox('input-post', 'preview-post', 'filename-post', 'post');

  document.getElementById('upload-reset-btn').addEventListener('click', resetUploadScreen);
  document.getElementById('analyze-btn').addEventListener('click', analyzeImages);
  document.getElementById('reset-analysis-btn').addEventListener('click', () => {
    resetUploadScreen();
    showScreen('upload');
  });

  document.querySelectorAll('.top-nav-tab').forEach(tab => {
    tab.addEventListener('click', () => showScreen(tab.dataset.tab));
  });

  updateConfidenceChart({
    labels: ['No damage', 'Minor damage', 'Major damage', 'Destroyed'],
    datasets: [
      { label: 'CNN', data: [0, 0, 0, 0] },
      { label: 'CNN+KNN', data: [0, 0, 0, 0] },
    ],
  }, 'both');
  loadMetrics();
}

initializeFrontend();
