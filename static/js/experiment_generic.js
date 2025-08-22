// static/js/experiment_generic.js
document.addEventListener('DOMContentLoaded', () => {
  const schema   = window.SCHEMA || [];
  const model    = window.MODEL_NAME || 'model';
  const form     = document.getElementById('exp-form');
  const results  = document.getElementById('results');

  // Build inputs (identical behaviour to your DT/KNN/Logistic pages)
  schema.forEach(field => {
    const wrapper = document.createElement('div');
    const label   = document.createElement('label');
    label.textContent = (field.label || field.name).replace(/_/g,' ').toUpperCase();
    wrapper.appendChild(label);

    let input;
    if (field.type === 'number') {
      input = document.createElement('input');
      input.type = 'number';
      input.name = field.name;

      if (field.min != null) input.min = field.min;
      if (field.max != null) input.max = field.max;

      // use schema step if present; otherwise allow any decimal
      input.step = (field.step !== undefined && field.step !== null) ? field.step : 'any';

      // optional helper text if you already pass placeholder
      if (field.placeholder) input.placeholder = field.placeholder;

        // IMPORTANT: do NOT default to min; use schema default if provided, else leave blank
      if (field.default !== undefined && field.default !== null && field.default !== '') {
        input.value = field.default;
      }

      // better mobile keypad for numbers/decimals
      input.inputMode = 'decimal';
    } else {
      // categorical → dropdown with original labels
      input = document.createElement('select');
      input.name = field.name;
      (field.options || []).forEach(opt => {
        const o = document.createElement('option');
        // allow {value,label} or plain string
        const val   = typeof opt === 'object' ? opt.value : opt;
        const text  = typeof opt === 'object' ? opt.label : opt;
        o.value = val; o.textContent = text;
        input.appendChild(o);
      });
    }

    wrapper.appendChild(input);
    // insert before the submit button (last element child is the button)
    form.insertBefore(wrapper, form.lastElementChild);
  });

  // Submit → predict
  form.onsubmit = e => {
    e.preventDefault();
    const payload = Object.fromEntries(new FormData(form));

    fetch(`/experiment/${encodeURIComponent(model)}/predict`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    })
    .then(r => r.json())
    .then(res => {
      // show prediction
      const predBlock = `<h2>Predicted class: ${res.prediction}</h2>`;
      let chartBlock  = '';
      if (Array.isArray(res.probabilities)) {
        chartBlock = `<div id="prob-chart" style="width:100%;height:300px;"></div>`;
      }
      results.innerHTML = `${predBlock}${chartBlock}`;

      // probability bar chart (same logic you use now)
      if (Array.isArray(res.probabilities)) {
        Plotly.newPlot('prob-chart', [{
          x: res.probabilities.map((_,i)=>`Class ${i}`),
          y: res.probabilities,
          type: 'bar'
        }], { margin:{t:30}, yaxis:{range:[0,1]} });
      }

      // Auto-scroll to results
      results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    })
    .catch(err => {
      results.textContent = 'Error: ' + err;
    });
  };
});
