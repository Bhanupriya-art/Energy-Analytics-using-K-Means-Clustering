# Global Energy Analytics – Clustering & Visualization

This project focuses on analyzing **global energy data** using **PySpark’s K-Means clustering** to uncover hidden patterns and groupings across multiple energy-related metrics. The clustered results are further visualized through an **interactive Tableau dashboard** for better interpretation and decision-making.

---

## Project Overview

- Performed **unsupervised machine learning (K-Means)** on large-scale energy data using PySpark.
- Applied **data preprocessing**, null handling, feature engineering, and **standardization**.
- Evaluated clustering quality using the **Silhouette Score**.
- Exported results and built a **Tableau dashboard** to visually explore energy clusters.

---

## Tech Stack

- **PySpark**
- **Apache Spark MLlib**
- **Python**
- **Tableau**
- **CSV Dataset**

---

## Workflow

1. Load global energy dataset using PySpark
2. Drop missing values for clean analysis
3. Automatically detect numeric columns
4. Assemble features using `VectorAssembler`
5. Standardize data with `StandardScaler`
6. Apply **K-Means clustering (k = 4)**
7. Evaluate model using **Silhouette Score**
8. Export clustered data for visualization
9. Build an interactive Tableau dashboard

---

## Model Performance

- **Algorithm:** K-Means Clustering
- **Number of Clusters:** 4
- **Evaluation Metric:** Silhouette Score
- **Feature Scaling:** StandardScaler (Mean & Std)

---

## Tableau Dashboard

The Tableau dashboard provides:
- Clear visualization of energy clusters
- Comparative analysis across regions and metrics
- Easy-to-understand insights derived from clustering results

*Dashboard file is included in the repository.*

<div style="display:flex; flex-wrap:wrap; gap:12px;">
<img  alt="Screenshot 2026-01-04 at 2 52 17 PM" src="https://github.com/user-attachments/assets/7f2d6c67-da2e-4918-bba3-dfed86cd8861" />
<img width="440" height="520" alt="Screenshot 2026-01-04 at 2 52 36 PM" src="https://github.com/user-attachments/assets/40a4c77d-83f9-4084-87f0-f952b4d4eaa8" />
<img width="440" height="520" alt="Screenshot 2026-01-04 at 2 52 48 PM" src="https://github.com/user-attachments/assets/71cc51ae-1947-4835-9637-b18d2c1af5b5" />
<img  alt="Screenshot 2026-01-04 at 2 53 11 PM" src="https://github.com/user-attachments/assets/30b1c0a5-f44e-48a2-b346-dd40089a1ed1" />
</div>

---

## Key Learnings

- Handling large datasets efficiently using PySpark
- Importance of feature scaling in clustering
- Practical application of unsupervised learning
- Transforming ML outputs into meaningful visual insights


⭐ If you find this project useful, feel free to star the repository!
