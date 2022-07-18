# Prediction of Defaulters in Banking organisations using different Machine learning techniques


## Overview of the project


The banking sector mainly works on money lending business, i.e. the more money they lend to people whom they can get good interest with timely repayment, the more revenue is for the banks. This comes to identify a major loop hole in the banking area, i.e. to identify people who are more likely to miss their repayment charges,
so that  they can take purposeful actions whether to remind them in person in advance or take some strict action to avoid delinquency.

There are two terms which are commonly used in banks - Delinquent and Default.

Delinquent : where a borrower is not repaying charges and is behind by certain months
Default : where a borrower has not been able to pay charges and is behind for a long period of months and is unlikely to repay the charges.

In this project, we aim to predict and identify such borrowers who are likely to default in the next two years with serious delinquency of having delinquent more than 90 days, using different machine learning techniques.

The dataset used for this purpose, will have general information about the borrowers such as age, Monthly Income, Dependents and the historical data such as what is the Debt Ratio, what ratio of amount is owed with respect to credit limit, and the no of times defaulted in the past one, two, three months.


## Objective

To build a machcine learning model using the inputs/attributes which are general profile and historical records of a borrower to predict whether one is likely to have serious delinquency in the next 2 years.
Also, train the given model using different ML techniques and comparing the accuracy and performaing error analysis, to identify the best model.


## Dataset

Dataset location : input/cs-test.csv , input/cs-training.csv

The dataset contains general information about the profiles and historical banking information. about the borrowers.


## Technologies Used

Python Libraries used :- pandas , matplotlib , seaborn , numpy , scikit-learn , Imblearn (for tackling class imbalance problem ), Shap and LIME (for model interpretability) , Keras for Neural Network(Deep Learning architecture)


## System Development

Reference to notebook : bank_deafulters_ml.ipynb

In this project, we perform an in-deth analysing and extensive EDA to build our machine learning model in order to achieve the best accuracy for our model.
We perform the Exploratory Data Analysis to understand how the data is distributed and what is the behavior of the inputs with respect to target variable which is 
 **SeriousDelinquencyin2Years**. Data preprocessing will be done based on how the values are distributed such as are there any data entry errors that needed to be removed, outlier treatment, which is necessary for certain algorithms, imputing missing values (if there are any).
 We split the dataset into the train and test dataset using Stratified Sampling to maintain the event rate across the different datasets so that a model can learn behavior from the training dataset and can predict with certain accuracy up to some on the unseen dataset.

**Class Imbalance Problem**

An imbalanced classification problem is an example of a classification problem where the distribution of examples across the known classes is biased or skewed. The distribution can vary from a slight bias to a severe imbalance where there is one example in the minority class for hundreds, thousands, or millions of examples in the majority class or classes.This maybe caused due to Biased Sampling or Measurement Errors.

Imbalanced classifications pose a challenge for predictive modeling as most of the machine learning algorithms used for classification were designed around the assumption of an equal number of examples for each class. This results in models that have poor predictive performance, specifically for the minority class. This is a problem because typically, the minority class is more important and therefore the problem is more sensitive to classification errors for the minority class than the majority class.


After preprocessing the data, we apply different machine learning techniques 

Here, are the most commonly used techniques to deal with Imbalanced Classes Problem in ML :-

* Random Under-Sampling : removing some observations of the majority class, but it can cause overfitting and poor generalization to the test dataset.
* Random Over-Sampling : adding more copies to the minority class.
* Random under-sampling with imblearn : **RandomUnderSampler** is a fast and easy way to under-sample the majority class(es) by randomly picking samples with or without replacement.
* Random over-sampling with imblearn : **RandomUnderSampler**  provides most naive strategy is to generate new samples by randomly sampling with replacement of the currently available samples
*  Under-sampling with Tomek links 
* Synthetic Minority Oversampling Technique (SMOTE) :

SMOTE algorithm works in 4 simple steps:

1. Choose a minority class as the input vector
2. Find its k nearest neighbors (k_neighbors is specified as an argument in the SMOTE() function)
3. Choose one of these neighbors and place a synthetic point anywhere on the line joining the point under consideration and its chosen neighbor
4. Repeat the steps until data is balanced


For the scope of the project, we implement the given techniques to solve this problem -Upsampling the minority class(default rate) , Downsampling the majority class(non defaulters),SMOTE.

<img width="971" alt="Screenshot 2022-07-19 at 1 36 24 AM" src="https://user-images.githubusercontent.com/25201417/179610116-2165a82c-28de-492c-bd02-5ba0ed9153fb.png">


In this project, we apply different ML techniques , cross vavlidation techniques , and different metrics to evaluate model performance, such as Precision, Recall , F1 Score , AUC & ROC for tranining our model  and achieve best results.


<img width="461" alt="Screenshot 2022-07-19 at 1 36 01 AM" src="https://user-images.githubusercontent.com/25201417/179610051-992d1c90-1892-4baa-a53b-db44787cb186.png">




## Integrating MLFoundry experiment tracking and model monitoring system

TrueFoundry's MLFoundry experiment tracking and model monitoring system combines the strengths of open-source tools such as MLFlow, Whylogs, and others. It comes with a shareable dashboard where we can keep track of our tests and model, among other things.  We create a client for the MLFoundry repository and assign a project name. To make experiment tracking easier, we assign different names for different experiments as well as different runs. It is a client-side library that allows users to log their experiments, models, metrics, data & features. This data is fed to TrueFoundry’s monitoring systems to generate informative dashboards and insights.







