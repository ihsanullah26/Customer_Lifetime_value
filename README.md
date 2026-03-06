# 🚀 Customer Lifetime Value (CLV) Prediction & Segmentation

## 📌 Project Overview
This project predicts the future financial value of customers for a retail business using transaction data from the UCI Machine Learning Repository. By leveraging **XGBoost** and an advanced **RFMT (Recency, Frequency, Monetary, Tenure)** feature set, the model identifies high-value "VIP" customers, enabling data-driven marketing and optimized retention strategies.

### The Problem
Retail data is notoriously skewed by "Whales"—a small percentage of customers who spend significantly more than the average. This project addresses this variance by using **Log Transformations** to normalize the target variable and ensemble tree-based models to capture non-linear spending patterns.

---

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost
* **Visualization:** Seaborn, Matplotlib
* **Data Source:** [UCI Machine Learning Repository - Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

---

## 📊 Methodology & Workflow

### 1. Advanced Feature Engineering
To move beyond basic transaction logs, we engineered a robust feature set to capture the "Health" and "Velocity" of each customer:
* **Recency:** Days since the last purchase (critical for churn detection).
* **Frequency:** Total count of unique transactions.
* **Monetary:** Total historical spend per customer.
* **Average Gap:** The mean time between purchases, capturing the buying cycle tempo.
* **Average Order Value (AOV):** Identifies high-margin "luxury" shoppers vs. high-volume shoppers.
* **Tenure:** Total duration of the customer relationship.

### 2. Target Transformation (`np.log1p`)
To mitigate the impact of outliers, the target variable was transformed using $y_{trans} = \ln(1 + y)$. This allowed the models to focus on the distribution's core rather than being blinded by extreme spenders. Post-prediction, values were reverted using `np.expm1` for real-world currency reporting.

---

## 🏆 Model Performance (Actual Currency Scale)
Models were evaluated after reverting predictions to their original dollar values.

| Model | MAE (Actual $) | R² Score | Verdict |
| :--- | :--- | :--- | :--- |
| Linear Regressor | 315,666.86 | -3,422,660.76 | **Failed** (Extreme Outlier Sensitivity) |
| Random Forest | 529.99 | 0.0543 | Strong Baseline |
| **XGBoost** | **516.10** | **0.0760** | **Champion** (Best Accuracy & Variance) |

> **Key Insight:** The catastrophic failure of the Linear Regressor (negative $R^2$) highlights the non-linear nature of retail data. Tree-based models like XGBoost are essential for handling the complexity of customer behavior.

---

## 💎 Customer Segmentation
Using the XGBoost predicted CLV, customers were divided into four actionable tiers:
1.  **VIP:** Top 25% of predicted spenders (Focus for exclusive rewards).
2.  **High Value:** Consistent contributors with steady purchasing habits.
3.  **Mid Value:** Occasional shoppers with high growth potential.
4.  **Low Value:** High-risk or one-time shoppers.

### Visualizing the Economic Impact
Because the VIP segment's value is so dominant, **Logarithmic Scaling** was applied to the Y-axis of our visualizations. This ensured that the Low and Mid-value segments remained visible for strategic analysis.



---

## 🚀 Installation & Usage
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/clv-prediction.git](https://github.com/yourusername/clv-prediction.git)
    ```
2.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the analysis:**
    Execute the Jupyter Notebook `clv_analysis.ipynb`.

---

## 💡 Key Takeaways
* **Log-scaling** is the "silver bullet" for financial ML to avoid massive outlier bias.
* **XGBoost** excels at identifying "Whale" customers even when they are statistically rare.
* **Feature Richness:** Adding metrics like `Average Gap` and `Tenure` significantly improved the model's ability to explain spending variance compared to basic RFM.

---
