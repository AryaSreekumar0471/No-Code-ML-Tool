// static/js/select_features.js
document.addEventListener('DOMContentLoaded', () => {
  const previewBox = document.getElementById('previewBox');

  document.querySelectorAll('.var-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const feat = btn.dataset.col;

      // 1. Clear & build a new canvas
      previewBox.innerHTML = `
        <h5 class="text-center mb-3">${feat}</h5>
        <canvas id="featChart"></canvas>
      `;
      const ctx = document.getElementById('featChart').getContext('2d');

      // 2. Fetch data (URL‐encoded)
      fetch(`/feature_distribution_data/${encodeURIComponent(feat)}`)
        .then(res => {
          if (!res.ok) throw new Error(res.statusText);
          return res.json();
        })
        .then(json => {
          if (json.type === 'numeric') {
            // histogram
            const labels = json.bins
              .slice(0,-1)
              .map((b,i) => `${b.toFixed(1)}–${json.bins[i+1].toFixed(1)}`);
            new Chart(ctx, {
              type: 'bar',
              data: {
                labels,
                datasets: [{ data: json.counts, backgroundColor: '#1877F2' }]
              },
              options: { responsive: true, plugins: { legend: { display: false } } }
            });
          } else {
            // categorical bar chart
            const labels = Object.keys(json.counts);
            const data   = Object.values(json.counts);
            new Chart(ctx, {
              type: 'bar',
              data: { labels, datasets: [{ data, backgroundColor: '#1877F2' }] },
              options: {
                responsive:true,
                plugins:{ legend:{display:false} },
                scales:{ x:{ ticks:{ maxRotation:90 } } }
              }
            });
          }
        })
        .catch(err => {
          previewBox.innerHTML +=
            `<p class="text-danger text-center mt-3">Error loading data: ${err.message}</p>`;
        });
    });
  });
});
// 2) Select-All + individual feature toggles
  const selectAllChk  = document.getElementById('selectAll');
  const featureChecks = Array.from(document.querySelectorAll('.feature-checkbox'));
  const targetSelect  = document.getElementById('targetSelect');
  const nextBtn       = document.getElementById('nextBtn');

  function updateNextButton() {
    const anyFeat = featureChecks.some(c=>c.checked);
    const hasTarg = !!targetSelect.value;
    nextBtn.disabled = !(anyFeat && hasTarg);
  }

  selectAllChk.addEventListener('change', () => {
  const sel = targetSelect.value;
  featureChecks.forEach(c => {
    // Only check if not the selected target
    if (c.value !== sel) c.checked = selectAllChk.checked;
    else c.checked = false;
  });
  updateNextButton();
});

  featureChecks.forEach(c=>c.addEventListener('change', () => {
    if (!c.checked) selectAllChk.checked = false;
    updateNextButton();
  }));

  // 3) Hide selected target from feature list
  targetSelect.addEventListener('change', () => {
    const sel = targetSelect.value;
    featureChecks.forEach(chk => {
      const row = chk.closest('.form-check');
      if (chk.value === sel) {
        chk.checked = false;
        row.style.display = 'none';
      } else {
        row.style.display = '';
      }
    });
    updateNextButton();
  });

  // 4) Submit on Next
    nextBtn.addEventListener('click', () => {
    const selectedTarget   = targetSelect.value;
    let selectedFeatures = featureChecks.filter(c=>c.checked).map(c=>c.value);
    // Always remove the target if it is accidentally included
    selectedFeatures = selectedFeatures.filter(f => f !== selectedTarget);

    fetch('/select_features', {
      method: 'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ features:selectedFeatures, target:selectedTarget })
    })
    .then(r=>r.ok?r.json():Promise.reject(r.statusText))
    .then(()=>window.location.href='/data_overview')
    .catch(e=>alert('Error: '+e));
  });
