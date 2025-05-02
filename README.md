
# 📚 Collaborative Filtering for Book Rating Prediction

## 📌 Project Summary
This project implements three collaborative filtering algorithms to predict how users would rate books they haven’t yet rated. The system uses explicit user rating data and evaluates the accuracy of predictions using **Mean Absolute Difference (MAD)** on a held-out test set.

The main objective is to implement, analyze, and evaluate collaborative filtering methods — not to benchmark or competitively compare them.

---

## 💡 Why Item-Item Collaborative Filtering?

Item-Item Collaborative Filtering (CF) was chosen over User-User CF for the following reasons:

- **Stability**: Item preferences (books) tend to change less frequently than user behavior, making item similarities more stable over time.
- **Scalability**: The number of books is smaller and more manageable than the number of users in this dataset.
- **Sparsity Handling**: Books generally receive more ratings than a typical user gives, allowing for more meaningful similarity computation between items.
- **Performance**: For large-scale systems like Amazon or Goodreads, item-based recommenders are often preferred due to faster lookups and better performance in real-time systems.

---

## ⚙️ Implementation Choices

- Used a **mean-centered user-item matrix** to normalize ratings and remove individual user bias.
- Computed **cosine similarity** between item vectors (books) using the Scikit-learn `cosine_similarity()` function.
- Developed a modular prediction function that selects the **top-k similar books** the user has rated and performs a **weighted average**.
- Extended functionality with:
  - **Bias correction** to adjust for systemic user/book rating shifts.
  - **Author-based fallback** to handle cold-start scenarios.
- Carefully ensured **test set consistency** by making sure all test users are also present in the training set.

---

## 📂 Files Included
- `code.ipynb` – Jupyter notebook with all code and evaluation.
- `Users.csv`, `Books.csv`, `Ratings.csv` – Input datasets provided.

---

## 🧠 Methods Implemented

### 1. Basic Item-Item Collaborative Filtering
- Core CF model using cosine similarity.
- Predictions made using ratings from top-k most similar books.
- Function: `predict_rating`

### 2. Item-Item CF with Bias Correction
- Adds per-user and per-book bias terms to account for rating habits.
- Improves accuracy in presence of skewed or inflated scores.
- Function: `predict_rating_with_bias`
- Evaluation: `evaluate_cf_with_bias`

### 3. Item-Item CF with Author Fallback
- In case of insufficient data, uses average rating from same author.
- Helpful for cold-start book scenarios.
- Function: `predict_rating_with_author_fallback`
- Evaluation: `evaluate_cf_hybrid`

---

## 📊 Evaluation Results

### MAD vs. Neighborhood Size (k)

**Basic CF**:
```
k=5    : 7.6772
k=10   : 7.7092
k=15   : 7.7157
k=20   : 7.7173
k=50   : 7.7077
k=100  : 7.6981
```

**Bias-Corrected CF**:
```
k=5    : 0.9157
k=10   : 0.8792
k=15   : 0.8612
k=20   : 0.8502
k=50   : 0.8283
k=100  : 0.8207
```

---

### MAD vs. Training Sample Ratio

**Basic CF (k=5)**:
```
60% Train: 7.6553
65% Train: 7.6600
70% Train: 7.6796
75% Train: 7.6772
80% Train: 7.6809
85% Train: 7.6951
90% Train: 7.7418
```

**Bias-Corrected CF (k=50)**:
```
60% Train: 0.8566
65% Train: 0.8481
70% Train: 0.8397
75% Train: 0.8283
80% Train: 0.8206
85% Train: 0.8104
90% Train: 0.8027
```

---

## ▶️ How to Run

### Requirements
```bash
pip install pandas numpy scikit-learn matplotlib tqdm
```

### Steps
1. Place all CSV files in the same directory.
2. Open `code.ipynb` in Jupyter or VS Code.
3. Run each section sequentially:
   - Data loading and cleaning
   - Similarity computation
   - Prediction and evaluation
   - Visualizations

---

## ✅ Features Used

| Feature        | Used? | Notes                                 |
|----------------|-------|----------------------------------------|
| `User-ID`      | ✅    | Used for user-item matrix              |
| `ISBN`         | ✅    | Book-level identifier                  |
| `Book-Rating`  | ✅    | Core of collaborative filtering        |
| `Book-Author`  | ✅    | Used for fallback in Method 3          |
| `Age`, `Location` | ❌ | Cleaned but not used in modeling       |

---

## 🧪 Future Improvements
- Add full content-based similarity (genre, title embeddings, etc.)
- Try matrix factorization (SVD/NMF) for latent features.
- Handle implicit feedback and optimize for ranking instead of rating.

---

## 👤 Author

**Mohit Saluru**  
Master's in Data Science  
University of Maryland, College Park  
May 2025
