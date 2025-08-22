document.addEventListener('DOMContentLoaded', () => {
  const trainBtn    = document.getElementById('trainBtn');
  const resultBox   = document.getElementById('resultBox');
  const metricsDiv  = document.getElementById('metrics');
  const plotButtons = document.getElementById('plotButtons');
  const experimentBtn = document.getElementById('experimentBtn');

  // NEW: download button + holder for last metrics
  const downloadBtn = document.getElementById('downloadResultsBtn');
  let lastMetrics = null;

  const loadingModal = new bootstrap.Modal(
    document.getElementById('loadingModal'),
    { backdrop: 'static', keyboard: false }
  );
  const plotModal = new bootstrap.Modal(
    document.getElementById('plotModal')
  );
  const plotBody = document.getElementById('plotBody');

  // Hide plot buttons initially
  plotButtons.style.display = 'none';
  plotButtons.classList.remove('d-flex');

  // TRAIN button logic
  trainBtn.addEventListener('click', () => {
    // reset UI
    resultBox.style.display = 'none';
    resultBox.style.opacity = '0';
    plotButtons.style.display = 'none';
    plotButtons.classList.remove('d-flex');
    experimentBtn.disabled = true;
    if (downloadBtn) downloadBtn.style.display = 'none';   // NEW: hide download button until new run
    loadingModal.show();

    fetch('/api/train_decision_tree', { method: 'POST' })
      .then(res => res.json())
      .then(data => {
        loadingModal.hide();
        // Show metrics
        metricsDiv.innerHTML = `
          <div><strong>Accuracy:</strong> ${data.accuracy} (±${data.accuracy_std})</div>
          <div><strong>Precision:</strong> ${data.precision}</div>
          <div><strong>Recall:</strong> ${data.recall}</div>
          <div><strong>F1-score:</strong> ${data.f1}</div>
        `;
        resultBox.style.display = 'block';
        setTimeout(() => { resultBox.style.opacity = '1'; }, 50);
        setTimeout(() => {
          resultBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 250);

        // Reveal plot buttons
        plotButtons.style.display = 'flex';
        plotButtons.classList.add('d-flex');
        experimentBtn.disabled = false;

        // NEW: store metrics & show download button
        lastMetrics = data;
        if (downloadBtn) downloadBtn.style.display = 'inline-block';
      })
      .catch(err => {
        loadingModal.hide();
        alert('Error during training: ' + err.message);
      });
  });

  // Generic helper to fetch JSON and draw with Plotly
  function showPlot(type, title) {
  // set the modal title & inject a placeholder div
  document.getElementById('plotTitle').innerText = title;
  plotBody.innerHTML = '<div id="plotContainer" style="width:100%;min-height:480px;"></div>';
  plotModal.show();

  fetch(`/plot/${type}`)
    .then(r => r.json())
    .then(obj => {
      Plotly.newPlot(
        'plotContainer',
        obj.data,
        obj.layout,
        { responsive: true, displayModeBar: true }
      );
    })
    .catch(err => {
      console.error(err);
      plotBody.innerHTML = '<div class="text-danger text-center my-4">Failed to load plot.</div>';
    });
}

  // Wire up buttons
  document.getElementById('showTreeBtn')
    .addEventListener('click', () => showPlot('decision_tree', 'Decision Tree'));
  document.getElementById('showFeatureBtn')
    .addEventListener('click', () => showPlot('feature_importance', 'Feature Importance'));
  document.getElementById('showCMBtn')
    .addEventListener('click', () => showPlot('confusion_matrix', 'Confusion Matrix'));

 
// train_decision_tree.js, bottom of the file:
// NEW
experimentBtn.addEventListener('click', () => {
  const model = experimentBtn.dataset.model || 'decision_tree';
  window.location.href = `/experiment/${model}`;
});


  // NEW: download results handler
  if (downloadBtn) {
    downloadBtn.addEventListener('click', () => {
      if (!lastMetrics) {
        alert('Please train the model first.');
        return;
      }
      fetch('/download_results', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ metrics: lastMetrics })
      })
      .then(res => {
        if (!res.ok) throw new Error('Failed to generate the file');
        return res.blob();
      })
      .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'decision_tree_results.xlsx';
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
      })
      .catch(err => alert('Download failed: ' + err.message));
    });
  }

});
