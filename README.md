# Credit-Card-Fruad-Project
This project untillizes machine learning models for detecting credit card fraud in financial organizations. The dataset was gotten from Kaggle, containing 8 attributes and 1,000,000 rows; implementation in Python.

## Abstract

Credit card fraud is an ever-growing problem, with billions of dollars lost in the financial industry. This study employs machine learning techniques for detecting credit card fraud within the financial sector; it also aimed to identify the most effective machine learning model for fraud detection. Using a dataset containing one million transactions obtained from Kaggle, the dataset has an imbalance distribution, and the oversampling technique "SMOTE" was applied to balance the distribution. Several algorithms were implemented: Random forest, Decision tree, K Nearest Neighbour, Logistic Regression, and Support Vector Machine. The Random Forest and Decision Tree model both demonstrated outstanding results, achieving the same accuracy of 100%, but Random Forest was shown to be the best by giving a lesser number of Type I errors of "1" and Type II errors of "2,"  making it the most reliable algorithm for this task. In contrast, with K Nearest    Neighbour with an accuracy of 100%, Logistic regression with an accuracy of 93%, and support vector machines with an accuracy of 95%, this model exhibits a higher misclassification error despite also showing high accuracy. The precision, recall, F1-score, and ROC-AUC were also used to evaluate all the models to ensure the models' effectiveness. This work also provides a comprehensive framework for detecting credit card fraud, highlighting the potential of machine learning to mitigate fraudulent activities in the financial sector. The project finishes by suggesting new research opportunities, such as investigating additional machine learning techniques and applying these models to diverse datasets to improve prediction accuracy.

## AIM AND OBJECTIVE
This dissertation aims to use machine learning technique to detect credit card fraud in financial sector.

### Objectives
1.	Literature review
2.	Methodology
3.	Implementation
4.	Results, findings, interpretations, and discussions
5.	Critical evaluation and conclusions

## RESEARCH QUESTION
1.	Which machine learning algorithms would be most effective in detecting credit card fraud ?
2.	What are the ethical implications of using machine learning for credit card fraud detection ?
3.	What is the cost-benefit analysis of implementing machine learning-based fraud detection systems for financial institutions ?


## DATA SOURCE

The dataset was retrieved from an open-source website, Kaggle.com, founded in 2010, which is a platform where people can find datasets, participate in competitions, and collaborate with others in the fields of machine learning and data science. The dataset consists of 8 attributes and 1,000,000 rows, which are all numeric. The attributes include "distance_from_home," which means the distance from home where the transaction happened, and "distance_from_last_transaction," which also means the distance from where the last transaction happened. The last observation is the class "fraud," which contains binary variables where “1” is a case of fraudulent transaction and “0” is not a case of fraudulent transaction.

Dataset : https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud/data

# Algorithm
1. Random Forest
2. Decision Tree
3. K Nearest Neighbour (KNN)
4. Logistic Regression
5. Support Vector Machine (SVM)

## LIMITATIONS

While this study uses machine learning techniques to detect credit card fraud, several limitations should be noted: First, the dataset consists of high-imbalance data, which may cause a bias model toward non-fraudulent transactions. The study relied on a single dataset sourced from Kaggle, which limits the generalizability of the findings.

The SVM model was not fully explored because of the runtime taking up to 1344 minutes to run one kernel, and there are 5 kernels in the algorithm, which requires a large processing power. Applying the sampling technique should be carefully examined toward the building of the KNN model because it could cause some shift in the picking of the nearest neighbour, making a lot of misclassification errors. Further, the risk of overfitting should be considered because the models may overfit to training data, leading to poor performance on unseen data.

Despite these limitations, this study provides a foundational framework for understanding and detecting credit card fraud, highlighting avenues for further investigation and refinement in the field of fraud detection and machine learning.

## CONCLUSION
In conclusion, this dissertation explores the efficacy of machine learning techniques in detecting credit card fraud in the financial sector. By implementing and evaluating several machine algorithms, such as random forest, decision tree, K-nearest neighbours, logistic regression and support vector machine, this study aimed to identify the most effective method for identifying fraudulent and non-fraudulent transactions.

This study reveals that the random forest algorithm is the best performing model, achieving an accuracy of 100% along with the lowest Type I of 1 and Type II errors of 6. This suggests that Random Forest, with its ensemble approach, is particularly well-suited for handling the complex patterns inherent in credit card fraud detection. The decision tree also came close to the same performance of Random Forest, given an accuracy of 100%. Although other models like K-Nearest Neighbours, Logistic Regression and support vector also performed well in terms of accuracy, they were less consistent in minimizing misclassification errors, highlighting the importance of selecting appropriate evaluation metrics beyond just accuracy. The support vector machine had a long run time of over 1388 minute on each kernel and still product a large mis classification error, which could be conclusion that it is not s suitable model for this data set.

The results of this study underscore the potential of machine learning to enhance fraud detection systems, providing financial institutions with tools that are both accurate and efficient. However, it became apparent that machine learning models are sensitive to the data's quality and the selected evaluation criteria. Problems like as data imbalance and overfitting were resolved using techniques like SMOTE and regularization. However, these difficulties highlight the importance of carefully model tuning and validation in practical scenarios.

In conclusion, this study examines the expanding knowledge on fraud detection by demonstrating the effectiveness of machine learning algorithms, specifically Random Forest, to mitigate credit card fraud. It offers a strong foundation for future study and practical application, with the ability to greatly decrease financial losses and improve the security of electronic transactions in the financial industry.




   
