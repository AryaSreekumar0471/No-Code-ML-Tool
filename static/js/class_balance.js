document.addEventListener('DOMContentLoaded', () => {
  // 1) Enable Bootstrap tooltips
  document.querySelectorAll('[data-bs-toggle="tooltip"]')
    .forEach(el => new bootstrap.Tooltip(el));

  // 2) Show/hide percent input
  const bothRadio    = document.getElementById('optBoth');
  const bothControls = document.getElementById('bothControls');
  document.getElementsByName('method')
    .forEach(radio => radio.addEventListener('change', () => {
      bothControls.style.display = bothRadio.checked ? 'block' : 'none';
    }));

  // 3) Init Chart.js from data-attribute
  const canvas = document.getElementById('balanceChart');
  const initialCounts = JSON.parse(canvas.dataset.counts);
  const ctx = canvas.getContext('2d');
  const chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: Object.keys(initialCounts),
      datasets: [{ 
        label: 'Count',
        data: Object.values(initialCounts),
        backgroundColor: 'rgba(24,119,242,0.7)'
      }]
    },
    options: { scales: { y: { beginAtZero: true } } }
  });

  document.getElementById('btnRebalance').addEventListener('click', () => {
    const method = document.querySelector('input[name="method"]:checked')?.value;
    if (!method) return alert('Select a method first.');
    const percent = document.getElementById('percentInput').value || 50;

    // 👉 Show loading indicator
    const loadingEl = document.getElementById('loading-indicator');
    if (loadingEl) loadingEl.style.display = 'block';

    fetch('/class-balance-data', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method, percent })
    })
    .then(res => res.json().then(payload => {
      if (!res.ok) throw new Error(payload.error || 'Unknown error');
      return payload;
    }))
    .then(json => {
      chart.data.labels = Object.keys(json.counts);
      chart.data.datasets[0].data = Object.values(json.counts);
      chart.update();
    })
    .catch(err => {
      alert('Error applying rebalance:\n' + err.message);
      console.error(err);
    })
    .finally(() => {
      //  Hide loading indicator
      if (loadingEl) loadingEl.style.display = 'none';
    });
  });

  // ---- MOVE THESE INSIDE ----
  // btnBack (if you have a Back button)
  const btnBack = document.getElementById('btnBack');
  if(btnBack){
    btnBack.addEventListener('click', () => {
      window.location.href = "{{ url_for('data_overview') }}";
    });
  }

  // btnNext
  document.getElementById('btnNext').addEventListener('click', function() {
    const folds = document.getElementById('cvFolds').value;
    const nextUrl = this.getAttribute('data-next-url');   // ← This is the fix!
    fetch('/set_cv_folds', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ folds })
    })
    .then(res => res.json())
    .then(json => {
      if (json.success) {
        window.location.href = nextUrl;
      } else {
        alert('Could not save CV folds, please try again.');
      }
    })
    .catch(err => {
      console.error(err);
      alert('Error saving CV folds');
    });
  });
});
