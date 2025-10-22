const input = document.getElementById("imageInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const results = document.getElementById("results");
const loading = document.getElementById("loading");

analyzeBtn.onclick = async () => {
  const file = input.files[0];
  if (!file) {
    alert("Please upload an image first!");
    return;
  }

  results.innerHTML = "";
  loading.classList.remove("hidden");

  const formData = new FormData();
  formData.append("image", file);

  try {
    const res = await fetch("/analyze", { method: "POST", body: formData });
    const data = await res.json();
    loading.classList.add("hidden");

    if (data.error) {
      results.innerHTML = `<p class='error'>${data.error}</p>`;
      return;
    }

    let html = "<h3>Detected:</h3><ul>";
    data.predictions.forEach(p => {
      html += `<li>${p.label} — ${p.score}%</li>`;
    });
    html += "</ul>";

    if (data.recommendations && data.recommendations.length > 0) {
      html += "<h3>Study Boost Recommendations 🍽️</h3><ul>";
      data.recommendations.forEach(r => {
        html += `<li><strong>${r.food}</strong>: ${r.advice} <em>(${r.confidence}% confidence)</em></li>`;
      });
      html += "</ul>";
    } else {
      html += "<p>No specific brain foods detected. Try retaking the photo!</p>";
    }

    results.innerHTML = html;

  } catch (err) {
    loading.classList.add("hidden");
    results.innerHTML = `<p class='error'>Error: ${err.message}</p>`;
  }
};
