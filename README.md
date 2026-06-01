Credit Scoring Business Understanding
Dataset

This project uses the Xente eCommerce transaction dataset
provided as part of the Bati Bank Credit Risk Modeling Challenge.

Source:
https://www.kaggle.com/datasets/brendah/xente-challenge

1.  Basel II and Model Interpretability

Basel II is a banking regulation that emphasizes accurate risk measurement, transparency, and accountability in lending decisions. Financial institutions must be able to explain how their models work, why specific features are used, how risk is calculated, and whether the decisions made by the model are fair and defensible.

Because of these requirements, credit risk models should be interpretable and well documented. Clear documentation allows regulators, auditors, and business stakeholders to understand and validate the model's behavior. This is especially important when loan approval or rejection decisions affect customers.

2.  Need for a Proxy Variable

The dataset does not contain a direct default label indicating whether a customer failed to repay a loan. Since supervised machine learning models require a target variable, a proxy variable must be created.

In this project, customer transaction behavior will be used to estimate risk. Customers with low engagement, low spending, and infrequent activity may be classified as high-risk customers through RFM analysis and clustering.

However, proxy variables introduce business risks. The proxy is only an estimate of default behavior and may incorrectly classify some customers. As a result, model predictions should be interpreted as risk estimates rather than confirmed default outcomes.

3.  Trade-Off Between Interpretability and Performance

Simple models such as Logistic Regression combined with Weight of Evidence (WoE) are highly interpretable. The influence of each feature on the prediction can be easily explained, making these models suitable for regulatory environments.

More advanced models such as Gradient Boosting or XGBoost often achieve higher predictive performance because they can capture complex patterns in the data. However, these models are more difficult to explain and may be viewed as black-box models.

In a regulated financial environment, organizations must balance predictive accuracy with interpretability. While high-performance models may improve risk prediction, interpretable models provide greater transparency and regulatory compliance.
