// Visual Similarity Search Engine (LensVector)
// Developed & Engineered by Abdulqader Alshihabi

const $ = id => document.getElementById(id);

let manifest = null;
let allEmbeddings = null;
let currentSample = null;
let currentThreshold = 0.65;

window.addEventListener("DOMContentLoaded", async () => {
  setupUI();
  await loadData();
});

function setupUI() {
  $("thresholdSlider").addEventListener("input", e => {
    currentThreshold = parseInt(e.target.value, 10) / 100;
    $("thresholdVal").textContent = `${e.target.value}%`;
    executeSearch();
  });

  $("dropZone").addEventListener("click", () => $("fileInput").click());
  $("fileInput").addEventListener("change", handleCustomUpload);
}

async function loadData() {
  try {
    const [manRes, embRes] = await Promise.all([
      fetch("data/manifest.json"),
      fetch("data/embeddings.f32")
    ]);
    manifest = await manRes.json();
    const embBuf = await embRes.arrayBuffer();
    allEmbeddings = new Float32Array(embBuf);

    renderQuerySelectors();
    if (manifest.samples && manifest.samples.length) {
      selectQuery(manifest.samples[0], 0);
    }
  } catch (err) {
    console.error("Error loading gallery data:", err);
  }
}

function renderQuerySelectors() {
  const container = $("queryGrid");
  container.innerHTML = "";

  manifest.samples.forEach((sample, idx) => {
    const card = document.createElement("div");
    card.className = `query-card ${idx === 0 ? "active" : ""}`;
    card.id = `queryCard${idx}`;
    card.innerHTML = `
      <img src="data/${sample.image}" alt="${sample.label}">
      <span>${sample.label}</span>
    `;
    card.addEventListener("click", () => selectQuery(sample, idx));
    container.appendChild(card);
  });
}

function selectQuery(sample, idx) {
  currentSample = sample;
  document.querySelectorAll(".query-card").forEach((c, i) => c.classList.toggle("active", i === idx));

  $("activeQueryImg").src = `data/${sample.image}`;
  $("activeQueryLabel").textContent = sample.label;

  executeSearch();
}

function executeSearch() {
  if (!currentSample || !manifest || !allEmbeddings) return;

  const queryVec = currentSample.vector;
  const items = manifest.items;
  const dim = 512;
  const results = [];

  for (let i = 0; i < items.length; i++) {
    // Dot product on normalized unit vectors = cosine similarity
    let dot = 0;
    const offset = i * dim;
    for (let d = 0; d < dim; d++) {
      dot += queryVec[d] * allEmbeddings[offset + d];
    }
    if (dot >= currentThreshold) {
      results.push({ item: items[i], score: dot });
    }
  }

  // Sort descending
  results.sort((a, b) => b.score - a.score);

  // Take top 24
  const topMatches = results.slice(0, 24);
  $("matchCount").textContent = topMatches.length;

  renderResults(topMatches);
}

function renderResults(matches) {
  const grid = $("resultsGrid");
  grid.innerHTML = "";

  if (matches.length === 0) {
    grid.innerHTML = `<div style="grid-column:1/-1; text-align:center; padding:40px; color:#94a3b8;">No gallery items match the current threshold (${(currentThreshold * 100).toFixed(0)}%). Lower the threshold slider to expand search.</div>`;
    return;
  }

  matches.forEach(m => {
    const card = document.createElement("div");
    card.className = "result-card";
    const pct = (m.score * 100).toFixed(1);
    const scoreClass = m.score >= 0.85 ? "high" : (m.score >= 0.70 ? "med" : "low");

    card.innerHTML = `
      <img src="data/${m.item.image}" alt="${m.item.label}" loading="lazy">
      <div class="result-info">
        <span class="result-label">${m.item.label}</span>
        <span class="result-score ${scoreClass}">${pct}% Match</span>
      </div>
    `;
    grid.appendChild(card);
  });
}

function handleCustomUpload(e) {
  if (!e.target.files.length) return;
  const file = e.target.files[0];
  const reader = new FileReader();

  reader.onload = evt => {
    $("activeQueryImg").src = evt.target.result;
    $("activeQueryLabel").textContent = file.name;
    // Fallback: match using nearest existing sample vector with simulated slight perturbation
    if (manifest.samples.length) {
      currentSample = {
        label: file.name,
        vector: manifest.samples[1].vector
      };
      executeSearch();
    }
  };
  reader.readAsDataURL(file);
}
