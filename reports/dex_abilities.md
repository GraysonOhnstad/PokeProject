# Dex generator: abilities model

5-fold family cross-validation (Gen 5 held out), Gen 3+ entries, species weighted equally. Phase-1 reference (shared heads): recall@5 16.3%; with kNN blend 18.7%.

| Metric | text-knn | text | oracle-up | pred-up |
|---|---:|---:|---:|---:|
| Ability-set recall@5 ↑ | 16.6% ± 1.3% | 17.0% ± 1.2% | 26.9% ± 3.8% | 17.2% ± 1.4% |
| Primary ability top-1 ↑ | 8.9% ± 1.1% | 10.7% ± 2.1% | 17.8% ± 3.0% | 9.3% ± 0.9% |
| Primary ability top-5 ↑ | 19.1% ± 1.5% | 22.6% ± 2.1% | 36.1% ± 4.1% | 21.6% ± 1.7% |
| Hidden ability top-5 ↑ (Gen 5+) | 8.1% ± 1.4% | 9.3% ± 2.3% | 11.3% ± 4.4% | 9.8% ± 3.6% |
