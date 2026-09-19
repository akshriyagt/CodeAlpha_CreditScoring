# CodeAlpha_CreditScoring

Machine Learning Internship – **Task 1: Credit Scoring Model** (CodeAlpha)

Predicts whether an individual is creditworthy using past financial data.

## Dataset
"Give Me Some Credit" (Kaggle) – 150,000 records, 6.7% defaulters.
Target column: `SeriousDlqin2yrs` (1 = risky, 0 = good).

## Approach
- **Feature engineering:** total late payments, 90-day-late flag, income per person, debt burden, missing-income flag, outlier capping
- **Models:** Logistic Regression, Decision Tree, Random Forest (class-weight balanced)
- **Metrics:** Precision, Recall, F1-Score, ROC-AUC, confusion matrix, feature importance
- **Split:** 80% train / 20% test (stratified)

## Run
```bash
pip install -r requirements.txt
python credit_scoring.py --data cs-training.csv --target SeriousDlqin2yrs
```

## Outputs
`results.csv`, `roc_curves.png`, `confusion_matrix.png`, `feature_importance.png`

## Results

| Model               | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|-----------|--------|----------|---------|
| Logistic Regression | 0.1816    | 0.6698 | 0.2858   | 0.8021  |
| Decision Tree       | 0.2051    | 0.7741 | 0.3242   | 0.8495  |
| Random Forest       | 0.5712    | 0.1561 | 0.2452   | 0.8442  |

## Conclusion

The dataset is highly imbalanced (only 6.7% risky customers), so accuracy was
not used; Precision, Recall, F1 and ROC-AUC were used instead.

Decision Tree gave the best ROC-AUC (0.85) and the highest Recall (0.77), so it
catches the most risky borrowers. Random Forest had a similar ROC-AUC (0.84) and
the highest Precision (0.57), but low Recall (0.16), meaning it misses many
defaulters. Logistic Regression was the simplest and most interpretable, but
scored lowest.

For a bank, missing a defaulter is costly, so Recall matters. The best choice
depends on the trade-off between rejecting good customers (Precision) and
approving risky ones (Recall).

Future work: threshold tuning, SMOTE for imbalance, and gradient boosting (XGBoost).