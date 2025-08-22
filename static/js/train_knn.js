document.addEventListener('DOMContentLoaded', () => {
  const trainBtn      = document.getElementById('trainBtn');
  const resultBox     = document.getElementById('resultBox');
  const metricsDiv    = document.getElementById('metrics');
  const plotButtons   = document.getElementById('plotButtons');
  const experimentBtn = document.getElementById('experimentBtn');

  // NEW: download button + holder for metrics
  const downloadResultsKnnBtn = document.getElementById('downloadResultsKnnBtn');
  let knnLastMetrics = null;

  const loadingModal = new bootstrap.Modal(
    document.getElementById('loadingModal'),
    { backdrop: 'static', keyboard: false }
  );
  const plotModal = new bootstrap.Modal(
    document.getElementById('plotModal')
  );
  const plotBody = document.getElementById('plotBody');

  plotButtons.style.display = 'none';

  trainBtn.addEventListener('click', () => {
    resultBox.style.display = 'none';
    resultBox.style.opacity = '0';
    plotButtons.style.display = 'none';
    experimentBtn.disabled = true;

    // NEW: hide download button on every fresh run
    if (downloadResultsKnnBtn) downloadResultsKnnBtn.style.display = 'none';

    loadingModal.show();

    fetch('/api/train_knn', { method: 'POST' })
      .then(res => res.json())
      .then(data => {
        loadingModal.hide();
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

        plotButtons.style.display = 'flex';
        experimentBtn.disabled = false;

        // NEW: store metrics & show download button
        knnLastMetrics = data;
        if (downloadResultsKnnBtn) downloadResultsKnnBtn.style.display = 'inline-block';
      })
      .catch(err => {
        loadingModal.hide();
        alert('Error during training: ' + err.message);
      });
  });

  function showPlot(endpoint, title) {
    document.getElementById('plotTitle').innerText = title;
    plotBody.innerHTML = '<div id="plotContainer" style="width:100%;min-height:480px;"></div>';
    plotModal.show();

    fetch(`/plot/${endpoint}`)
      .then(r => r.json())
      .then(obj => {
        Plotly.newPlot('plotContainer', obj.data, obj.layout, { responsive: true });
      })
      .catch(err => {
        console.error(err);
        plotBody.innerHTML = '<div class="text-danger text-center my-4">Failed to load plot.</div>';
      });
  }

  document.getElementById('showErrBtn')
    .addEventListener('click', () => showPlot('error_vs_k', 'Error vs. k'));
  document.getElementById('showBoundBtn')
    .addEventListener('click', () => showPlot('decision_boundary', '2D Decision Boundary'));
  document.getElementById('showCMBtn')
    .addEventListener('click', () => showPlot('knn_confusion_matrix', 'Confusion Matrix'));

  experimentBtn.addEventListener('click', () => {
    const model = experimentBtn.dataset.model || 'K-Nearest Neighbour';
    window.location.href = `/experiment/${model}`;
  });


  // NEW: download handler
  if (downloadResultsKnnBtn) {
    downloadResultsKnnBtn.addEventListener('click', () => {
      if (!knnLastMetrics) {
        alert('Please train the model first.');
        return;
      }
      fetch('/download_results_knn', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ metrics: knnLastMetrics })
      })
      .then(res => {
        if (!res.ok) throw new Error('Failed to generate the file');
        return res.blob();
      })
      .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'knn_results.xlsx';
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
      })
      .catch(err => alert('Download failed: ' + err.message));
    });
  }
});
