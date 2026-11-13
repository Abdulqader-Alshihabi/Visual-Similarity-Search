# Visual Similarity Search Engine (LensVector) with OpenAI CLIP

[![Live Demo](https://img.shields.io/badge/Demo-Live_GitHub_Pages-0ea5e9?style=for-the-badge&logo=googlechrome&logoColor=white)](https://abdulqader-alshihabi.github.io/Visual-Similarity-Search/)
[![Colab](https://img.shields.io/badge/Notebook-Open_in_Colab-f59e0b?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/Abdulqader-Alshihabi/Visual-Similarity-Search/blob/main/Image_Finder_CLIP.ipynb)
[![Model](https://img.shields.io/badge/Model-OpenAI_CLIP_ViT--B%2F32-818cf8?style=for-the-badge&logo=openai&logoColor=white)](https://github.com/Abdulqader-Alshihabi/Visual-Similarity-Search)
[![Vector Dimension](https://img.shields.io/badge/Vector_Space-512_Dimensions-34d399?style=for-the-badge)](https://github.com/Abdulqader-Alshihabi/Visual-Similarity-Search)
[![License](https://img.shields.io/badge/License-MIT-gray?style=for-the-badge)](LICENSE)

An interactive zero-shot multimodal image search and retrieval engine. Powered by OpenAI's Contrastive Language-Image Pre-Training (CLIP ViT-B/32) vision encoder, executing sub-15ms client-side nearest neighbor retrieval over a 2,000-image embedding space directly inside the browser using vectorized cosine similarity.

---

## Vector Search & Multimodal Theory

Traditional keyword search fails when indexing diverse visual concepts without exhaustive manual tagging. Visual similarity search circumvents this by projecting unconstrained images into a shared continuous semantic vector space:

```
[ Query Image (e.g. Koala / Sports Car) ]
                   |
                   v
+-------------------------------------------+
|  OpenAI CLIP Vision Transformer (ViT-B/32)|
|  - 12-layer Vision Transformer            |
|  - 32x32 pixel patch projection           |
|  - Multimodal semantic embedding          |
+-------------------------------------------+
                   |
                   v  [ 512-dimensional float32 vector q ]
       Normalized: q / ||q||_2
                   |
                   v
+-------------------------------------------+
|  Client-Side Vector Search Engine         |
|  - Computes Dot Product: sim(q, v_i)      |
|  - 2,000 Gallery Embeddings (v_1..v_2000) |
|  - Cosine Similarity: -1.0 to +1.0        |
+-------------------------------------------+
                   |
                   v
[ Ranked Gallery Matches: Cosine Similarity % (e.g. 94.8% Match) ]
```

Because all vectors are pre-normalized to unit sphere length ($\|v\|_2 = 1.0$), the cosine similarity simplifies to an ultra-fast raw dot product:

$$	ext{CosineSimilarity}(q, v) = \sum_{d=1}^{512} q_d \cdot v_d$$

---

## Key System Highlights

- **Zero-Shot Semantic Alignment:** Accurately groups related conceptual matches (e.g., matching a koala query to other marsupials, arboreal mammals, and eucalyptus wildlife) without explicit category labels.
- **Ultra-Fast In-Memory Search:** Searches the full 2,000 $	imes$ 512 vector index in less than 15 milliseconds directly in JavaScript using `Float32Array` SIMD operations.
- **Zero Cloud Costs:** The entire embedding index (~4.1 MB binary file) is delivered via static CDN and cached locally in the user's browser.
- **Interactive Similarity Threshold:** Real-time slider (50% to 95%) dynamically filtering nearest neighbors based on semantic confidence.
- **Custom Image Matching:** Allows users to drop personal images to query the vector gallery.

---

## Technical Specifications

| Parameter | Specification | Details |
| :--- | :--- | :--- |
| **Model** | OpenAI CLIP (ViT-B/32) | Contrastive Vision Transformer |
| **Patch Size** | 32 $	imes$ 32 pixels | Spatial visual tokenizer |
| **Embedding Dimension** | 512 float32 values | Dense continuous latent space |
| **Gallery Index Size** | 2,000 images | Subset sampled from Tiny ImageNet |
| **Memory Footprint** | ~4.1 MB binary | Highly compact binary format |
| **Query Search Speed** | **< 15 ms** | In-browser client CPU execution |

---

## Repository Structure

```
Visual-Similarity-Search/
├── docs/                               # Static GitHub Pages deployment
│   ├── index.html                      # Interactive visual search web application
│   ├── style.css                       # Modern dark slate & indigo stylesheet
│   ├── app.js                          # Vector math, dot product engine, query UI
│   ├── data/
│   │   ├── manifest.json               # Metadata for 2,000 gallery images & benchmark queries
│   │   ├── embeddings.f32              # Binary 2,000 x 512 float32 embedding matrix
│   │   └── images/                     # Gallery image assets
│   ├── search.js                       # Cosine similarity helper
│   └── model-worker.js                 # Web worker infrastructure
├── Image_Finder_CLIP.ipynb             # Executed PyTorch embedding extraction notebook
└── README.md                           # Comprehensive engineering documentation
```

---

## Quickstart & Local Execution

### 1. Launch Web Search Engine
```bash
git clone https://github.com/Abdulqader-Alshihabi/Visual-Similarity-Search.git
cd Visual-Similarity-Search
python -m http.server 8000 --directory docs
```
Navigate to `http://localhost:8000` in any web browser.

### 2. Python Embedding Extraction & Verification
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

pip install torch torchvision ftfy regex tqdm pillow

python -c "
import numpy as np
embeddings = np.fromfile('docs/data/embeddings.f32', dtype=np.float32).reshape(-1, 512)
print(f'Successfully loaded gallery matrix: {embeddings.shape[0]} images across {embeddings.shape[1]} dimensions.')
"
```

---

## Author & Engineering Credentials

**Abdulqader Alshihabi**  
Computer Science Graduate · Islamic University of Madinah (GPA: 4.62 / 5.00, First Class Honors)  
Software & AI Engineer specializing in Deep Learning, Multimodal Vision, and Scalable Full-Stack Systems.

- **Portfolio:** [https://abdulqader-alshihabi.github.io/](https://abdulqader-alshihabi.github.io/)
- **GitHub:** [@Abdulqader-Alshihabi](https://github.com/Abdulqader-Alshihabi)
- **LinkedIn:** [https://www.linkedin.com/in/Abdulqader-Alshihabi/](https://www.linkedin.com/in/Abdulqader-Alshihabi/)
- **Email:** [abdulqader.moh.2003@gmail.com](mailto:abdulqader.moh.2003@gmail.com)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
