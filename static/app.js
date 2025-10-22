const uploadInput = document.getElementById("imageInput");
const cameraInput = document.getElementById("cameraInput");
const uploadBtn = document.getElementById("uploadBtn");
const cameraBtn = document.getElementById("cameraBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const results = document.getElementById("results");
const loading = document.getElementById("loading");
const previewArea = document.getElementById("preview-area");

let selectedFile = null;

uploadBtn.onclick = () => uploadInput.click();
cameraBtn.onclick = () => cameraInput.click();

uploadInput.onchange = (e) => handleFile(e.target.files[0]);
cameraInput.onchange = (e) => handleFile(e.target.files[0]);

function handleFile(file) {
  if (!file) return;
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewArea.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
  };
  reader.readAsDataURL(file);
}

analyzeBtn.onclick = async () => {
  if (!selectedFile) {
    alert("Please choose or take a photo first!");
    return;
  }

  results.innerHTML = "";
  loading.classList.remove("hidden");

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const res = await fetch("/analyze", { method: "POST", body: formData });
    const data = await res.json();
    loading.classList.add("hidden");

    if (data.error) {
      results.innerHTML = `<p style="color:#ff7777;">${data.error}</p>`;
      return;
    }

    let html = "<h3>Detected foods:</h3><ul>";
    (data.predictions || []).forEach(p => {
      html += `<li>${p.label} — ${(p.score*100).toFixed(1)}%</li>`;
    });
    html += "</ul>";

    if (data.recommendations && data.recommendations.length) {
      html += "<h3>Study Boost Tips 🍽️</h3><ul>";
      data.recommendations.forEach(r => {
        html += `<li><strong>${r.food}</strong>: ${r.advice}</li>`;
      });
      html += "</ul>";
    } else {
      html += "<p>No study-food items detected 😅</p>";
    }
    results.innerHTML = html;
  } catch (err) {
    loading.classList.add("hidden");
    results.innerHTML = `<p style="color:#ff7777;">Error: ${err.message}</p>`;
  }
};
