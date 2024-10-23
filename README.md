# Credit-Card-Fruad-Project
Credit Card Fraud Detection Using Machine Learning Techniques

## Abstract

Credit card fraud is an ever-growing problem, with billions of dollars lost in the financial industry. This study employs machine learning techniques for detecting credit card fraud within the financial sector; it also aimed to identify the most effective machine learning model for fraud detection. Using a dataset containing one million transactions obtained from Kaggle, the dataset has an imbalance distribution, and the oversampling technique "SMOTE" was applied to balance the distribution. Several algorithms were implemented: random forest, decision tree, K nearest neighbour, logistic regression, and support vector machine. The random forest and decision tree model both demonstrated outstanding results, achieving the same accuracy of 100%, but random forest was shown to be the best by giving a lesser number of Type I errors of "1" and Type II errors of "2,"  making it the most reliable algorithm for this task. In contrast, with K nearest neighbour with an accuracy of 100%, logistic regression with an accuracy of 93%, and support vector machines with an accuracy of 95%, this model exhibits a higher misclassification error despite also showing high accuracy. The precision, recall, F1-score, and ROC-AUC were also used to evaluate all the models to ensure the models' effectiveness. This work also provides a comprehensive framework for detecting credit card fraud, highlighting the potential of machine learning to mitigate fraudulent activities in the financial sector. The project finishes by suggesting new research opportunities, such as investigating additional machine learning techniques and applying these models to diverse datasets to improve prediction accuracy.

## AIM AND OBJECTIVE
This dissertation aims to use machine learning technique to detect credit card fraud in financial sector.

### Objectives
1.	Literature review
2.	Methodology
3.	Implementation
4.	Results, findings, interpretations, and discussions
5.	Critical evaluation and conclusions

## RESEARCH QUESTION
•	Which machine learning algorithms would be most effective in detecting credit card fraud ?
•	What are the ethical implications of using machine learning for credit card fraud detection?
•	What is the cost-benefit analysis of implementing machine learning-based fraud detection systems for financial institutions?


## DATA SOURCE

The dataset was retrieved from an open-source website, Kaggle.com, founded in 2010, which is a platform where people can find datasets, participate in competitions, and collaborate with others in the fields of machine learning and data science. The dataset consists of 8 attributes and 1,000,000 rows, which are all numeric. The attributes include "distance_from_home," which means the distance from home where the transaction happened, and "distance_from_last_transaction," which also means the distance from where the last transaction happened. The last observation is the class "fraud," which contains binary variables where “1” is a case of fraudulent transaction and “0” is not a case of fraudulent transaction.

## LIMITATIONS

While this study uses machine learning techniques to detect credit card fraud, several limitations should be noted: First, the dataset consists of high-imbalance data, which may cause a bias model toward non-fraudulent transactions. The study relied on a single dataset sourced from Kaggle, which limits the generalizability of the findings.

The SVM model was not fully explored because of the runtime taking up to 1344 minutes to run one kernel, and there are 5 kernels in the algorithm, which requires a large processing power. Applying the sampling technique should be carefully examined toward the building of the KNN model because it could cause some shift in the picking of the nearest neighbour, making a lot of misclassification errors. Further, the risk of overfitting should be considered because the models may overfit to training data, leading to poor performance on unseen data.

Despite these limitations, this study provides a foundational framework for understanding and detecting credit card fraud, highlighting avenues for further investigation and refinement in the field of fraud detection and machine learning.



   
