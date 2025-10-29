# Machine Learning: Advanced

Summary: Today we will help you master advanced tasks involved in machine learning in Python.

💡 [Tap here](https://new.oprosso.net/p/4cb31ec3f47a4596bc758ea1861fb624) **to leave your feedback on the project**. It's anonymous and will help our team make your educational experience better. We recommend completing the survey immediately after the project.

## Contents

1. [Chapter I](#chapter-i) \
    1.1. [Foreword](#foreword)
2. [Chapter II](#chapter-ii) \
    2.1. [Instructions](#instructions)
3. [Chapter III](#chapter-iii) \
    3.1. [Specific instructions for the day](#specific-instructions-for-the-day)
4. [Chapter IV](#chapter-iv) \
    4.1. [Exercise 00: Regularization](#exercise-00-regularization)
5. [Chapter V](#chapter-v) \
    5.1. [Exercise 01: Gridsearch](#exercise-01-gridsearch)
6. [Chapter VI](#chapter-vi) \
    6.1. [Exercise 02: Metrics](#exercise-02-metrics)
7. [Chapter VII](#chapter-vii) \
    7.1. [Exercise 03: Ensembles](#exercise-03-ensembles)
8. [Chapter VIII](#chapter-viii) \
    8.1. [Exercise 04: Pipelines and OOP](#exercise-04-pipelines-and-oop)
    
   
## Chapter I

### Foreword

There is real data science, and then there is data science for competitions. The difference is similar to that between a porter and a weightlifter. A porter may perform poorly in heavy lifting competitions, and a weightlifter may underperform in a real job. The same is true for data science. In companies, you need a broader skill set, including soft skills, for example. You need to understand the business and focus on profit or the organization's other goals. In competitions, however, you simply need to excel at achieving the target metrics. You could create an extremely advanced machine learning model that outperforms your next competitor's model by 0.00001, and no one would care how long it takes to make predictions, how interpretable the model is, or how many computational resources it requires. In business, of course, all of these criteria matter.

Nevertheless, you can use competitions to improve your skills. The most popular platform is [Kaggle](https://www.kaggle.com/). There are many public datasets and competitions. You can try to win a prize, but you will most likely only earn experience, which is not bad at all. This can be useful for building your portfolio. You can organize a team and learn from each other. You can also improve your collaboration skills. The platform allows you to pursue either the machine learning or data exploration track, depending on your preference.

Think of it as a great tool for continued growth in the field, but keep in mind that real life requires more than building machine learning models and optimizing hyperparameters.

## Chapter II

### Instructions

* Use this page as your only reference. Do not pay attention to rumors or speculation about how to prepare your solution.
* Here and throughout, we use Python 3 as the only correct version of Python.
* The python files for python exercises (module01, module02, module03) must have the following block at the end: `if __name__ == ‘__main__’`.
* Pay attention to the permissions of your files and directories.
* To be assessed your solution must be in your GIT repository.
* Your solutions will be evaluated by your peers in the bootcamp.
* You should not leave any other files in your directory other than those explicitly specified in the exercise instructions. It is recommended that you modify your .gitignore to avoid any accidents.
* Your solution must be in your GIT repository for evaluation. Always push only to the develop branch! The master branch will be ignored. Work in the src directory.
* When you need to get precise output in your programs, it is forbidden to display a precalculated output instead of performing the exercise correctly.
* Have a question? Ask your neighbor on the right. If that fails, try your neighbor on the left.
* Your reference materials are your peers, the internet, and Google.
* Read the examples carefully. They may require information that is not specified elsewhere in the subject.
* May the Force be with you!

## Chapter III

### Specific instructions for the day

* Use the Jupyter notebook to work with your code.
* For each major subtask in the list of exercises, your ipynb file should have an H2 heading to help your peers easily navigate your code.
* No imports are allowed except those mentioned in the "Authorized Functions" section of the title block of each exercise.
* You can use any built-in function as long as it is not prohibited by the exercise.
* Save and load all required data in the `data/` subfolder.
* `scikit-learn` (0.23.1) is the library that you need for all the machine learning tasks.
* `tqdm` (4.46.1) is the library that you need for tracking progress.

## Chapter IV

### Exercise 00: Regularization

- Turn-in directory: `ex00/`.
- Files to turn in: `regularization.ipynb`.
- Allowed functions: no restrictions.

In the previous day, you tried solving different types of machine learning problems using various algorithms, which helped you grasp the fundamentals. Today, you will try more advanced techniques.

Let's start with regularization. In its broadest sense, regularization is a technique that prevents a model from overfitting. Regarding logistic regression, we have a formula with different Xs (features) and coefficients. Regularization penalizes coefficients that are too large, making the formula more robust and better able to handle unknown data in the future. L1 regularization sets some coefficients to zero. Thus, this method may be helpful for feature selection when there are many features and the number needs to be reduced. L2 regularization does not set the weights to zero, but it may reduce their magnitude. We cannot say, "Use only L2 or L1 regularization". In machine learning, there are few silver bullets. Usually, you need to try many different approaches with your dataset to find the best fit.

Regarding trees and forests, regularization is connected to the parameters that affect the number of cases in the leaves. If your tree is so dense that each leaf contains only one sample, chances are your tree has overfitted the training dataset.

To prevent that, you can adjust parameters such as `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_leaf_nodes`.

Many algorithms have different regularization parameters. Our goal is not to cover them all. You just need to know that they exist in case you need them and want to understand how they work for a specific algorithm.

In this exercise, you will experiment with some of these parameters. The dataset and task will remain the same: predict the weekday for each commit using the following data: uid, lab name, number of trials, and the hour of the commit.

What you need to do is fully described in the [notebook](https://drive.google.com/file/d/1-bx3kLOrZhe6eGRUj6187SVCglHUyqW5/view).

## Chapter V

### Exercise 01: Gridsearch

- Turn-in directory: `ex01/`.
- Files to turn in: `gridsearch.ipynb`.
- Allowed functions: no restrictions.

We're sure you're tired of manually iterating through the different parameters of various models. You probably think there should be a way to automate the process. Yes, there is: GridSearch.

Specify the range of values for the parameters you want to optimize, then put them into GridSearchCV. GridSearchCV will try all of them, calculate the metrics on cross-validation, and provide the best combination of parameters, as well as the overall results of its mini-research. Cool, right?

What you need to do describe in the [notebook](https://drive.google.com/file/d/1qH7LmNIrzkH3ViC1aPfnHT5rVKBcEFrQ/view).

## Chapter VI

### Exercise 02: Metrics

- Turn-in directory: `ex02/`.
- Files to turn in: `metrics.ipynb`.
- Allowed functions: no restrictions.

Is 90% accuracy a good result? It's actually not easy to say. Imagine two unbalanced classes: 95% of the samples belong to the first class, and 5% belong to the second. Accuracy will be worse than with a naive classifier when making predictions using the most popular class. This is not a fantasy. This situation is common in anti-fraud tasks, for example. The number of fraud cases is significantly lower than the number of normal cases. Is this a bad metric? It is simple to understand, and you can use it to compare different models within a task. However, this metric can be misleading when comparing results to a model from another task. Additionally, it does not provide much information about errors. You only know how many there are. But what kind?

Some other metrics can answer this question. They all come from the [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix). The first is precision. It is the number of correctly predicted samples of one class divided by the total number of predictions of that class. For example, imagine that we predicted 10 days as weekends, but only seven of them actually were. The precision would be 0.7.

The second is recall. It is the number of correctly predicted samples of one class divided by the true number of that class. Again, imagine that we predicted 10 days as weekends, but only 7 of them were actually weekends. In the dataset, there were 20 weekends. The recall would be 0.35 (7/20).

Precision is useful when we want to show an ad with 16+ content. We want to make precise predictions. Recall can be useful for identifying terrorists. We want to find all of them, no matter how many civilians experience inconvenience along the way. There is also a metric that combines both: the harmonic mean F1 score. Use it when you need to optimize both.

Also, there is the [ROC-curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic). When making predictions, you typically use probabilities. The final classification is made by comparing them to the threshold. For example, if the probability of it being a weekend on a given day is 0.2 and the threshold is 0.5, then it is not a weekend. 

However, if the threshold were changed to 0.1, the same sample would be predicted as a weekend. Now, imagine that we calculate the recall for each threshold and the number of working days predicted as weekends divided by the actual number of working days. We can plot both of these values to get the ROC curve. The higher the curve, the better. However, comparing curves can be inconvenient. That is why we use another metric, the area under the curve (AUC). 

Precision, recall, and AUC are useful for comparing the performance of different models from different tasks. They also tell us something about errors. Everything we discussed here was related to binary classification. However, with some adjustments, it can be used for multiclass and multilabel classification. For example, you can read about it [here](https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2), for example.

What you need to do is fully described in the [notebook](https://drive.google.com/file/d/18FA62WxDMYUhhk1dWqb8dhcG3kDw6Xf1/view).

## Chapter VII

### Exercise 03: Ensembles

- Turn-in directory: `ex03/`.
- Files to turn in: `ensembles.ipynb`.
- Allowed functions: no restrictions.

As you know, a random forest is an ensemble of many different trees. However, you can actually create an ensemble from any type of model. In this exercise, you will try three approaches: a voting classifier, a bagging classifier, and a stacking classifier. Who knows? Maybe it will help you improve the quality of your predictions.

What you need to do is fully described in the [notebook](https://drive.google.com/file/d/1z_DTgLXCNw_uTTjkWMzzFFVJc03xSSc2/view).

## Chapter VIII

### Exercise 04: Pipelines and OOP

- Turn-in directory: `ex04/`.
- Files to turn in: pipelines.ipynb.
- Allowed functions: no restrictions.

While trying to solve the problem, you performed many different actions: you prepared the data, tried different models and metrics, optimized their hyperparameters, and tried different kinds of ensembles. Your code is probably scattered across different notebooks, requiring a lot of scrolling, and it probably looks chaotic. 

In this final exercise of the day, you will clean up your code and make it more organized. Why is this important? In real life, you may want to share it with your colleagues or include it in your portfolio, or you may want it for your own future convenience. If you return to that code several months from now, you may think, "Who made that mess?"

In this exercise, you will apply the object-oriented programming (OOP) approach to data analysis. The first part of your notebook will contain only imports, classes, and methods. The second part will be your "main program". You will work with the initial data and go through most of the previous steps.

What you need to do is fully described in the [notebook](https://disk.360.yandex.ru/d/jXE4MK0G5C5PKQ).
