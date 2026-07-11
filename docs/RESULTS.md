# Current Results

## Predictive ceiling

| Score | AUC |
| --- | ---: |
| True structural risk probability | 0.701 |
| XGBoost | 0.684 |

The low-looking AUC is expected for the current weak-signal structural model.
Correctly specified logistic models and a probability forest did not reveal a
substantially higher exploitable ceiling.

## Ordering alone does not solve the problem

| Method | Kendall tau | Top-5 recovery | PBI | Mass within 2 hops |
| --- | ---: | ---: | ---: | ---: |
| Exact TreeSHAP | 0.522 | 0.60 | 1.082 | 83.1% |
| Matched ordinary interventional SHAP | 0.506 | 0.60 | 1.051 | 81.6% |
| DAG-asymmetric SHAP | 0.528 | 0.60 | 1.051 | 81.2% |

Paired bootstrap intervals include zero for the differences in rank recovery,
PBI, POA, and proximal mass between matched ordinary and DAG-asymmetric SHAP.
Restricting feature order therefore does not, by itself, support a causal
recovery claim.

## Structural prototype

| Metric | Structural prototype | Ordering-only DAG-asymmetric |
| --- | ---: | ---: |
| Kendall tau | 0.794 | 0.528 |
| Spearman rho | 0.932 | 0.652 |
| Top-5 recovery | 1.00 | 0.60 |
| PBI | -0.113 | 1.051 |
| POA | -0.0226 | 0.210 |
| Mass within 2 hops | 38.1% | 81.2% |

The frozen truth places 42.1% of importance within two hops. The prototype is
slightly too upstream but far closer than the predictive estimators. These
numbers remain provisional until the structural estimator is scaled,
bootstrapped, and repeated across simulation seeds.

## Pedagogic stress test

The app also contains a deliberately dramatic mediator/proxy example. It is
useful for teaching the difference between prediction and intervention, but it
is not NASA evidence and is never used as the primary scientific result.
