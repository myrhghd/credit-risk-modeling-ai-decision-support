# Home Credit 2024 Source And Feature Catalog Summary

This catalog is modeling-oriented, not exhaustive profiling. It uses raw PostgreSQL
schemas, `feature_definitions.csv`, existing Parquet metadata, and small `LIMIT`
samples. It does not create indexes, physical staging copies, or engineered features.

## Feature Definition Coverage

- Unique raw source columns: 471
- Raw columns represented in `feature_definitions.csv`: 464
- Raw columns without supplied descriptions: 7
- Definitions not present in ingested training tables: 1

## Logical Tables

| table | depth | rows | columns | business information | historical grain | justified feature-engineering categories |
| --- | --- | --- | --- | --- | --- | --- |
| applprev_1 | depth 1 | 6,525,979 | 41 | Previous Home Credit application or contract history. | repeated records identified by case_id, num_group1 | age/recency/duration transformations, amount totals, ratios, and utilization-style summaries, delinquency/payment behavior summaries, min/mean/max/latest summaries for numeric history, per-case counts and recency-aware aggregations, stable applicant/person categorical encodings, status/category frequency and latest-value summaries |
| applprev_2 | depth 2 | 14,075,487 | 6 | Previous Home Credit application or contract history. | repeated records identified by case_id, num_group1, num_group2 | min/mean/max/latest summaries for numeric history, per-case counts and recency-aware aggregations, status/category frequency and latest-value summaries |
| base | base | 1,526,659 | 5 | Application outcome anchor containing one row per case, the binary target, decision date, and time index. | one row per case or case-level snapshot | age/recency/duration transformations, direct typed predictors and missingness indicators |
| credit_bureau_a_1 | depth 1 | 15,940,537 | 79 | External credit bureau history from source A. | repeated records identified by case_id, num_group1 | age/recency/duration transformations, amount totals, ratios, and utilization-style summaries, credit-history burden and contract-status summaries, delinquency/payment behavior summaries, min/mean/max/latest summaries for numeric history, per-case counts and recency-aware aggregations, status/category frequency and latest-value summaries |
| credit_bureau_a_2 | depth 2 | 188,298,452 | 19 | External credit bureau history from source A. | repeated records identified by case_id, num_group1, num_group2 | credit-history burden and contract-status summaries, delinquency/payment behavior summaries, min/mean/max/latest summaries for numeric history, per-case counts and recency-aware aggregations, status/category frequency and latest-value summaries |
| credit_bureau_b_1 | depth 1 | 85,791 | 45 | External credit bureau history from source B. | repeated records identified by case_id, num_group1 | age/recency/duration transformations, amount totals, ratios, and utilization-style summaries, credit-history burden and contract-status summaries, delinquency/payment behavior summaries, min/mean/max/latest summaries for numeric history, per-case counts and recency-aware aggregations, status/category frequency and latest-value summaries |
| credit_bureau_b_2 | depth 2 | 1,286,755 | 6 | External credit bureau history from source B. | repeated records identified by case_id, num_group1, num_group2 | delinquency/payment behavior summaries, min/mean/max/latest summaries for numeric history, per-case counts and recency-aware aggregations, status/category frequency and latest-value summaries |
| debitcard_1 | depth 1 | 157,302 | 6 | Debit card history linked to cases. | repeated records identified by case_id, num_group1 | age/recency/duration transformations, min/mean/max/latest summaries for numeric history, per-case counts and recency-aware aggregations, status/category frequency and latest-value summaries |
| deposit_1 | depth 1 | 145,086 | 5 | Deposit/account history linked to cases. | repeated records identified by case_id, num_group1 | age/recency/duration transformations, amount totals, ratios, and utilization-style summaries, min/mean/max/latest summaries for numeric history, per-case counts and recency-aware aggregations, status/category frequency and latest-value summaries |
| other_1 | depth 1 | 51,109 | 7 | Other case-linked historical signals. | repeated records identified by case_id, num_group1 | amount totals, ratios, and utilization-style summaries, min/mean/max/latest summaries for numeric history, per-case counts and recency-aware aggregations, status/category frequency and latest-value summaries |
| person_1 | depth 1 | 2,973,991 | 37 | Person/application participant information. | repeated records identified by case_id, num_group1 | age/recency/duration transformations, amount totals, ratios, and utilization-style summaries, min/mean/max/latest summaries for numeric history, per-case counts and recency-aware aggregations, stable applicant/person categorical encodings, status/category frequency and latest-value summaries |
| person_2 | depth 2 | 1,643,410 | 11 | Person/application participant information. | repeated records identified by case_id, num_group1, num_group2 | min/mean/max/latest summaries for numeric history, per-case counts and recency-aware aggregations, stable applicant/person categorical encodings, status/category frequency and latest-value summaries |
| static_0 | depth 0 | 1,526,659 | 168 | Application-level/static applicant fields available at or near the decision snapshot. | one row per case or case-level snapshot | age/recency/duration transformations, amount totals, ratios, and utilization-style summaries, credit-history burden and contract-status summaries, delinquency/payment behavior summaries, direct typed predictors and missingness indicators, stable applicant/person categorical encodings |
| static_cb_0 | depth 0 | 1,500,476 | 53 | Static credit-bureau-derived fields available at the decision snapshot. | one row per case or case-level snapshot | age/recency/duration transformations, credit-history burden and contract-status summaries, delinquency/payment behavior summaries, direct typed predictors and missingness indicators |
| tax_registry_a_1 | depth 1 | 3,275,770 | 5 | Tax registry source A fields linked to cases. | repeated records identified by case_id, num_group1 | age/recency/duration transformations, amount totals, ratios, and utilization-style summaries, min/mean/max/latest summaries for numeric history, per-case counts and recency-aware aggregations, status/category frequency and latest-value summaries |
| tax_registry_b_1 | depth 1 | 1,107,933 | 5 | Tax registry source B fields linked to cases. | repeated records identified by case_id, num_group1 | age/recency/duration transformations, amount totals, ratios, and utilization-style summaries, min/mean/max/latest summaries for numeric history, per-case counts and recency-aware aggregations, status/category frequency and latest-value summaries |
| tax_registry_c_1 | depth 1 | 3,343,800 | 5 | Tax registry source C fields linked to cases. | repeated records identified by case_id, num_group1 | age/recency/duration transformations, delinquency/payment behavior summaries, min/mean/max/latest summaries for numeric history, per-case counts and recency-aware aggregations, status/category frequency and latest-value summaries |

## Leakage Notes

`target` is the label and must be excluded from predictors. `date_decision`,
`WEEK_NUM`, and `MONTH` are temporal metadata suitable for validation design and
time-aware transformations, but not ordinary unconstrained predictors until the
temporal strategy is chosen. Other fields marked as leakage candidates are based
only on supplied descriptions containing decision/default/target language and need
manual review before modeling.
