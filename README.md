# Finding the Relevancy of Contextual Data in Online Shopper Purchasing Intentions

Final project for CSC 321 (Intro to Data Mining and Machine Learning). Tests
whether contextual features (browser, OS, region, month, visitor type) add
predictive value over behavioral features alone (page views, time on page,
bounce/exit rates) when classifying whether a shopping session generates
revenue.

**Author:** Theo Scola

## Files

- [`TS_CSC_321_Final.ipynb`](./TS_CSC_321_Final.ipynb) — full analysis:
  exploratory correlation heatmaps, feature selection, model comparison, and
  statistical testing.

Data is pulled directly from a public raw GitHub URL
([`ML-Final-Data`](https://github.com/tscola24/ML-Final-Data)) at runtime, so
the notebook runs anywhere without path edits or Colab-specific setup.

## Approach

1. **Dataset:** UCI Online Shoppers Purchasing Intentions dataset — 12,330
   sessions, 17 features, imbalanced (84.5% no-revenue).
2. **Feature screening:** correlation heatmaps for each contextual feature
   group (OS/browser, visitor type, date/month/special-day, region) against
   revenue, to decide what's worth keeping before modeling.
3. **Feature selection:** dropped contextual features with no observed
   correlation, kept and encoded the few with signal (November as a
   Black-Friday/Cyber-Monday proxy, visitor type).
4. **Modeling:** two logistic regression pipelines — one with the retained
   contextual features, one behavioral-only — plus a ZeroR baseline, each
   evaluated with 10-fold cross-validation.
5. **Significance testing:** paired t-tests between all three models rather
   than comparing accuracy numbers directly.

## Results

- Both logistic regression models (~87-88% accuracy) statistically
  outperformed the ZeroR baseline (~84%).
- No statistically significant difference between the contextual and
  behavioral-only models (paired t-test, p > 0.05).

## Conclusion

Contextual data doesn't improve prediction accuracy for this task — the
behavioral-only model performs identically to the one with contextual
features added. By Occam's razor, the simpler behavioral-only model is
preferable: same accuracy, less dimensionality, and no need to collect or
maintain contextual data in production. The main exploratory finding worth
keeping is the November revenue spike (holiday shopping) and a small edge
for new visitors over returning ones; most other contextual signals
(browser, OS, region, weekends, special-day proximity) showed no
relationship to revenue at all.
