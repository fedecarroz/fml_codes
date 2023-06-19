# Foundation of Machine Learning codes

## Folder structure and description
---

```bash
│   # CLUSTERING
│   # Clustering-related classes built from scratch.
├── clustering
│   # KMeans class
│   ├── kmeans.py # KMeans class
│   └── kmedoids.py # KMedoids class
│   
│   # DATASETS
├── datasets
│   ├── advertising.csv
│   ├── cancer.csv
│   ├── cardio.csv
│   ├── glass.csv
│   ├── houses.csv
│   ├── insurance.csv
│   ├── petrol_consumption.csv
│   ├── student_scores.csv
│   ├── transfusion.csv
│   └── wine.csv
│   
│   # ESTIMATORS
│   # sci-kit learn "equivalent" estimators built from scratch.
│   # 
│   # FBGD => Full-Batch Gradient Descent
│   # SGD => Stochatsic Gradient Descent
│   # MBGD => Mini-Batch Gradient Descent
│   # OLS => Ordinary Least Squares (Normal Equation method)
│   # clf => classification task
│   # reg => regression task
├── estimators
│   │   # linear regression classes
│   ├── linear_regression 
│   │   ├── fbgd_linear_regression.py 
│   │   ├── mbgd_linear_regression.py
│   │   ├── ols_linear_regression.py
│   │   └── sgd_linear_regression.py
│   │   # logistic regression classes
│   ├── logistic_regression
│   │   ├── fbgd_logistic_regression.py
│   │   ├── mbgd_logistic_regression.py
│   │   └── sgd_logistic_regression.py
│   │   # neural netowrk classes
│   └── neural_network 
│       ├── fbgd_clf_neural_network.py
│       └── fbgd_reg_neural_network.py
│   
│   # HELPERS
│   # other helpful classes or methods built from scratch.
├── helpers
│   ├── grid_search_cv.py
│   ├── normalization.py
│   └── sigmoid.py
│   
│   # MAIN
│   # Main files in which supervised and unsupervised learning tasks are performed.
│   # Classes built from scratch and sci-kit library has been adopted.
├── main
│   ├── anomaly.py
│   ├── clustering.py
│   ├── grid_search_log_reg.py
│   ├── houses.py
│   ├── nn_clf_main.py
│   ├── nn_reg_main.py
│   ├── pca.py
│   ├── petrol_consumption.py
│   └── student_scores.py
│   
│   # METRICS
│   # Model evaluation utilities built from scratch.
└── metrics
    ├── classification_metrics.py
    ├── metrics_evaluation.py
    ├── regression_metrics.py
    └── roc.py
```

## Credits
---
Thanks to Federico Carrozzino, Giovanni Silvestri and Paolo Masciullo