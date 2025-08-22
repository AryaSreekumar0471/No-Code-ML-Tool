document.addEventListener('DOMContentLoaded', () => {
  const slider  = document.getElementById('ratioSlider');
  const hidden  = document.getElementById('ratioInput');
  const display = document.getElementById('countDisplay');

  const updateCounts = () => {
    const testPct    = parseInt(slider.value, 10);    // slider = test%
    const trainPct   = 100 - testPct;                 // train% = 100 - test%

    hidden.value     = trainPct;                      // submit train% directly

    const trainCount = Math.round(TOTAL_ROWS * trainPct / 100);
    const testCount  = TOTAL_ROWS - trainCount;
    display.textContent = 
      `Train: ${trainCount} rows | Test: ${testCount} rows`;
  };

  // initialize & live update
  updateCounts();
  slider.addEventListener('input', updateCounts);
});
