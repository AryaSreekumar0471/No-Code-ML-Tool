document.addEventListener('DOMContentLoaded', () => {
  const dropzone     = document.getElementById('dropzone');
  const fileInput    = document.getElementById('fileInput');
  const browseBtn    = document.getElementById('browseBtn');
  const uploadBtn    = document.getElementById('uploadBtn');
  const errorMsg     = document.getElementById('errorMsg');
  const progressDiv  = document.getElementById('progressContainer');
  const progressPct  = document.getElementById('progressPercent');
  const fileNameLbl  = document.getElementById('fileName');
  const tickIcon     = document.getElementById('tickIcon');
  const crossIcon    = document.getElementById('crossIcon');
  const previewBtn   = document.getElementById('previewBtn');
  let selectedFile   = null;
  let xhr            = null;

  ['dragenter','dragover','dragleave','drop'].forEach(evt => {
    dropzone.addEventListener(evt, e => {
      e.preventDefault(); e.stopPropagation();
    });
  });

  dropzone.addEventListener('dragover', () => dropzone.classList.add('dragover'));
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));

  dropzone.addEventListener('drop', e => {
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });

  browseBtn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });

  function handleFile(file) {
    errorMsg.textContent = '';
    if (!file.name.toLowerCase().endsWith('.csv')) {
      errorMsg.textContent = 'only csv file supported';
      selectedFile = null;
      return;
    }
    selectedFile = file;
    dropzone.querySelector('#dz-msg-1').textContent = file.name;
  }

  uploadBtn.addEventListener('click', () => {
    errorMsg.textContent = '';
    if (!selectedFile) {
      errorMsg.textContent = 'no file chosen';
      return;
    }

    tickIcon.classList.remove('active');
    crossIcon.style.opacity = '0.3';
    progressPct.textContent = '0%';
    fileNameLbl.textContent = selectedFile.name;
    previewBtn.disabled = true;

    progressDiv.style.display = 'flex';
    progressDiv.style.justifyContent = 'center';
    progressDiv.style.alignItems = 'center';

    xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append('dataset', selectedFile);

    xhr.open('POST', '/upload_file');
    xhr.upload.addEventListener('progress', e => {
      if (e.lengthComputable) {
        const pct = Math.round((e.loaded / e.total) * 100);
        progressPct.textContent = `${pct}%`;
      }
    });

    xhr.onload = () => {
      if (xhr.status === 200) {
        progressPct.textContent = '100%';
        tickIcon.classList.add('active');
        previewBtn.disabled = false;
      } else {
        errorMsg.textContent = 'Upload failed';
      }
    };

    xhr.send(form);
  });

  crossIcon.addEventListener('click', () => {
    if (xhr && xhr.readyState !== 4) {
      xhr.abort();
    } else if (selectedFile) {
      fetch('/delete_file', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({filename: selectedFile.name})
      });
    }

    progressDiv.style.display = 'none';
    previewBtn.disabled = true;
    selectedFile = null;
    fileInput.value = '';
    dropzone.querySelector('#dz-msg-1').textContent = 'Drag and Drop a file';
  });


// grab the modal element
const previewModalEl = document.getElementById('previewModal');
// listen for when it hides
previewModalEl.addEventListener('hidden.bs.modal', () => {
  // wipe out the ?preview and the hash
  history.replaceState(null, '', '/upload');
});



  previewBtn.addEventListener('click', () => {
  // reload the page, generating `preview` on the server,
  // and include a #previewModal hash so we know to pop it open on _this_ load.
  window.location.href = '/upload?preview=true#previewModal';
});
});
