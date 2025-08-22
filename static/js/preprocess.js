document.addEventListener('DOMContentLoaded', () => {
  // 1) Grab all the buttons & containers
  const btnCat    = document.getElementById('btnCat');
  const btnBox    = document.getElementById('btnBox');
  const btnMiss   = document.getElementById('btnMiss');
  const plotBox   = document.getElementById('plotBox');
  const introText = document.getElementById('introText');
  const chartWrapper = document.getElementById('chartWrapper');
  const procBtn   = document.getElementById('procBtn');
  const procMsg   = document.getElementById('procMsg');
  const nextBtn   = document.getElementById('nextBtn');
  const stepsForm = document.getElementById('stepsForm');

  // === NEW: controls for enabling/disabling buttons ===
  const downloadBtn = document.getElementById('downloadBtn');

  function setDownloadDisabled(disabled) {
    if (!downloadBtn) return;
    if (disabled) {
      downloadBtn.classList.add('disabled');
      downloadBtn.removeAttribute('href');
      downloadBtn.setAttribute('aria-disabled', 'true');
    } else {
      downloadBtn.classList.remove('disabled');
      downloadBtn.removeAttribute('aria-disabled');
    }
  }

  function updateProcBtn() {
    const anyChecked = stepsForm.querySelectorAll('input[type=checkbox]:checked').length > 0;
    procBtn.disabled = !anyChecked;
  }

  // show/hide encode radios
  const chkEncode  = document.getElementById('stepEncode');
  const encOptions = document.getElementById('encOptions');
  chkEncode.addEventListener('change', () => {
    encOptions.classList.toggle('hidden', !chkEncode.checked);
  });

  // show/hide outlier radios
  const chkOutliers = document.getElementById('stepOutliers');
  const outOptions  = document.getElementById('outOptions');
  chkOutliers.addEventListener('change', () => {
    outOptions.classList.toggle('hidden', !chkOutliers.checked);
  });

  // show/hide scale radios
  const chkScale     = document.getElementById('stepScale');
  const scaleOptions = document.getElementById('scaleOptions');
  chkScale.addEventListener('change', () => {
    scaleOptions.classList.toggle('hidden', !chkScale.checked);
  });

  // 2) Enable the action buttons
  [btnCat, btnBox, btnMiss, procBtn].forEach(btn => btn.disabled = false);

  // watch all checkboxes to update the Preprocess button state
  stepsForm.querySelectorAll('input[type=checkbox]').forEach(cb => {
    cb.addEventListener('change', updateProcBtn);
  });

  // initial state
  updateProcBtn();
  setDownloadDisabled(true);

  // 3) View Categorical Countplots
  btnCat.onclick = () => {
    introText.style.display = 'none';
    chartWrapper.style.display = 'block';
    procMsg.textContent = '';
    fetch('/categorical_count_data')
      .then(res => res.json())
      .then(json => {
        const data = json.data || {};
        plotBox.innerHTML = '';
        if (!Object.keys(data).length) {
          plotBox.innerHTML = `<p class="no-data">No categorical variables found in this dataset.</p>`;
          return;
        }
        Object.entries(data).forEach(([col, counts]) => {
          const ctn = document.createElement('div'); ctn.className = 'mb-4';
          const title = document.createElement('h6'); title.textContent = col;
          ctn.appendChild(title);
          const canvas = document.createElement('canvas');
          ctn.appendChild(canvas);
          plotBox.appendChild(ctn);
          new Chart(canvas.getContext('2d'), {
            type: 'bar',
            data: {
              labels: Object.keys(counts),
              datasets: [{ label: col, data: Object.values(counts), backgroundColor: '#1877F2' }]
            },
            options: {
              responsive: true,
              plugins: { legend: { display: false } },
              scales: { x: { ticks: { maxRotation: 90 } } }
            }
          });
        });
      });
  };

  // 4) View Box Plot
  btnBox.onclick = () => {
    introText.style.display = 'none';
    chartWrapper.style.display = 'block';
    procMsg.textContent = '';
    plotBox.innerHTML = '';
    const img = document.createElement('img');
    img.src = '/boxplot';
    img.alt = 'Box‐and‐Whisker Plots';
    img.style.maxWidth = '100%';
    img.style.marginTop = '1rem';
    plotBox.appendChild(img);
  };

  // 5) View Missing Data Chart
  btnMiss.onclick = () => {
    introText.style.display = 'none';
    chartWrapper.style.display = 'block';
    procMsg.textContent = '';
    fetch('/missing_data_data')
      .then(res => res.json())
      .then(json => {
        plotBox.innerHTML = '';
        const counts = json.data;
        const canvas = document.createElement('canvas');
        plotBox.appendChild(canvas);
        new Chart(canvas.getContext('2d'), {
          type: 'bar',
          data: {
            labels: Object.keys(counts),
            datasets: [{ label: 'Missing count', data: Object.values(counts), backgroundColor: '#1877F2' }]
          },
          options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: { x: { ticks: { maxRotation: 90 } } }
          }
        });
      });
  };

  // 6) Run Preprocessing Steps
  procBtn.onclick = () => {
    procMsg.textContent = '';

    // Only grab the checked CHECKBOXES, not radios
    const steps = Array.from(
      stepsForm.querySelectorAll('input[type=checkbox]:checked')
    ).map(i => i.value);

    // Guard: require at least one step
    if (steps.length === 0) {
      procMsg.textContent = 'Select at least one preprocessing step.';
      return;
    }

    // Disable download while processing
    setDownloadDisabled(true);

    procMsg.textContent = 'Preprocessing...';
    const payload = { steps };
    if (steps.includes('encode'))   payload.encode_method  = document.querySelector('input[name=encode_method]:checked')?.value;
    if (steps.includes('scale'))    payload.scale_method   = document.querySelector('input[name=scale_method]:checked')?.value;
    if (steps.includes('outliers')) payload.outlier_method = document.querySelector('input[name=outlier_method]:checked')?.value;

    fetch('/run_preprocessing', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    .then(res => {
      if (!res.ok) return res.json().then(j => { throw new Error(j.error || res.statusText); });
      return res.json();
    })
    .then(json => {
      procMsg.textContent = 'Preprocessing completed';

      // summary
      const logsDiv = document.getElementById('summaryLogs');
      logsDiv.innerHTML = '';
      json.summary.forEach(line => {
        const p = document.createElement('p'); p.textContent = line;
        logsDiv.appendChild(p);
      });

      // preview
      const previewDiv = document.getElementById('previewDiv');
      previewDiv.innerHTML = json.preview;

      // download link (enable now)
      const dl = document.getElementById('downloadBtn');
      dl.href = `/download_cleaned/${json.cleaned_file}`;
      setDownloadDisabled(false);

      document.getElementById('resultBox').style.display = 'block';
      nextBtn.disabled = false;
    })
    .catch(err => {
      console.error(err);
      procMsg.textContent = '❌ ' + err.message;
      setDownloadDisabled(true);
    });
  };  // ← **This closes procBtn.onclick**

  // 7) Next → button
  nextBtn.onclick = () => {
    window.location.href = '/select_features';
  };

}); // ← **This closes DOMContentLoaded**
