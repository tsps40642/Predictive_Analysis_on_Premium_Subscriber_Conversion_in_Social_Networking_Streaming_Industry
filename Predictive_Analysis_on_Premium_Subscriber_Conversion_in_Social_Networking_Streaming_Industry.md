Predictive Analysis on Premium Subscriber Conversion in Social
Networking Streaming Industry
================

Please note that some description parts are adjusted after rendering.

# GROUP PART – technical document

Table of contents: 1. Things to Notice before Analysis 1. Problem
defining and our overall rationale to solve it 2. EDA before Prediction:
(1) due to the imbalanced dataset, oversampling to train the model is
needed (2) not very strong positive correlations exist; we will fit the
model later to check whether there are severe multicollinearity using
Variance Inflation Factor (VIF) 3. Normalization: min-max normalization
4. Before Feature Selection: with / without oversampling + no feature
selection 5. Feature Selection: Filter Approach 6. Model Fitting and
Performance Evaluation: (1) fit the models with the filtered data set
(2) 10-Folds Cross-Validation with oversampled training set within each
folds (2) select the final model based on AUC (3) generate dashboard of
threshold and their corresponding precision and recall_p 7. Summary and
other analysis results 8. Translate analyzing results into business
solutions outline 9. Appendix: more model tuning and selection

## Things to Notice before Analysis

### Don’t use Accuracy in Imbalanced dataset; Use AUC, Precision and Recall_p

From the EDA later we will know that, the proportion of the 2 levels in
response variable is severely imbalanced, indicating we shouldn’t use
accuracy as our evaluation standard since it will conclude incorrect
results.

Thus, our priority of model selection is based on AUC, then precision
and recall_p. However, we are going to be not so restricted: if there’s
no really big difference in those metrices, we prefer more explainable
model in business point of view.

### Understanding the scope of our analysis: prediction task, not causal analysis

Note that all the data are originally non-subscribers. It is essential
to emphasize our goal for understanding which people would be likely to
convert from free users to premium subscribers in the next 6 month
period if they are targeted by our promotion campaign. We care about
correlation and can’t say “they turn into premium users DUE TO our
promotion” since it’s NOT an causal problem which statistical method we
have now cannot solve.

So please notice the description here: we are NOT going to use CAUSE,
DUE TO…

## Problem defining and our overall rationale to solve it

### General goal for analysis

To get a deeper understanding of which people would be likely to convert
from free users to premium subscribers in the next 6 month period if
they are targeted by our promotion campaign.

### Specific goal for analysis:

1.  train model to conduct prediction task, then select the final one
    based on AUC, precision, and recall_p.
2.  we need to do EDA before moving on to further analysis, and we are
    going to combine analysis from that as well as model selection to
    provide data support for business solutions.
3.  we will translate analyzing results into business solutions outline,
    and more detailed business strategies will be presented in our
    managerial document.

## EDA before Prediction

``` r
setwd('C:/Users/Yvonne/Desktop/UMN Courses/6131')
library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library(ggplot2)
library(caret)
```

    ## Loading required package: lattice

``` r
library(pROC)
```

    ## Type 'citation("pROC")' for a citation.

    ## 
    ## Attaching package: 'pROC'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     cov, smooth, var

``` r
library(ROSE) 
```

    ## Loaded ROSE 0.0-4

``` r
xyzdata <- read.csv('XYZData.csv')
```

Check missing values: no missing values

``` r
sum(is.na(xyzdata == TRUE)) # no missing values
```

    ## [1] 0

Check class proportion: imbalanced data
![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->
Due to the severe imbalanced data, oversampling to train the model is
needed.

Check the correlation among variables to get a conceptual understanding
of the dataset.

    ## corrplot 0.92 loaded

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->
No significant negative correlation but some positive correlations we
might want to notice later.

The followings are some reasons we consider why that positive
correlation happened: 1. age / avg_friend_age: people generally tends to
have friends with similar age range. 2. friend_cnt / friend_country_cnt:
a person with more friends tends to have more friends from more
different countries 3. friend_cnt / subscriber_friend_cnt: a person with
more friends tends to have more friends that are subscribers.

Check relationship between response and each predictor.
![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-5-2.png)<!-- -->![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-5-3.png)<!-- -->![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-5-4.png)<!-- -->![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-5-5.png)<!-- -->![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-5-6.png)<!-- -->![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-5-7.png)<!-- -->![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-5-8.png)<!-- -->
Variables seems to have pattern, and please note that this is a little
bit subjective judgement: (\>\>\>\>\> means we consider the variable
acts(or ranges) more differently in adopter_1 and adopter_0 compared to
others, i.e. more possible patterns) delta_shouts \>\>\>\>\>
delta_playlist delta_posts \>\>\> delta_lovedTracks \>\>\>\>\>
delta_songListened \>\>\>\>\> delta_subscriber_friend_cnt
delta_avg_friend_age \>\>\>\>\> delta_avg_friend_male \>\>\>
delta_friend_cnt \>\>\>\>\> posts lovedTracks \>\>\>\>\> shouts
\>\>\>\>\> playlists songListened \>\>\>\>\> subscriber_friend_cnt
friend_country_cnt avg_friend_age friend_cnt

And we have more limited countries without changing status and more
males than females.

Another thing we want to mention is that, we guess the performance of
model fitting might not be so ideal since we can’t observe clear pattern
in EDA, so we might focus on finding a model that can get the most
senses in business perspective rather than “torchering” data to get as
much as information we want. We will definitely do adjustment to find
better one, we are saying that although numerically thinking the
performance might not be perfect, yet how we can apply that in business
strategy is more important.

## Normalization

Implement min-max normalization before predicting.

``` r
normalize = function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# And transfer the response into factors for the two datasets.
xyzdata$adopter <- as.factor(xyzdata$adopter)

# use the mutate_at() to specify the indexes of columns needed normalization
# we can and need to normalize both binary and numerical data, except for adopter and user_id
xyzdata_normalized <- xyzdata %>% mutate_at(c(2:26), normalize)

# we drop the user_id since it' just an index
xyzdata_normalized_drop_user_id <- xyzdata_normalized[, -1]

# use createDataPartition() fo split the training and testing dataset for w. delta data
train_rows <- createDataPartition(y = xyzdata_normalized_drop_user_id$adopter, p = 0.75, list = FALSE)
xyzdata_normalized_drop_user_id_train <- xyzdata_normalized_drop_user_id[train_rows, ]
xyzdata_normalized_drop_user_id_test <- xyzdata_normalized_drop_user_id[-train_rows, ]
```

## Before Feature Selection: with / without oversampling + no feature selection

### Logistic regression: no oversampling, no feature selection

``` r
fit_log <- glm(adopter ~ ., data = xyzdata_normalized_drop_user_id_train, family = binomial)
summary(fit_log)
```

    ## 
    ## Call:
    ## glm(formula = adopter ~ ., family = binomial, data = xyzdata_normalized_drop_user_id_train)
    ## 
    ## Coefficients:
    ##                              Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)                  -6.20997    1.92362  -3.228 0.001245 ** 
    ## age                           1.64869    0.40415   4.079 4.52e-05 ***
    ## male                          0.38669    0.07024   5.505 3.68e-08 ***
    ## friend_cnt                  -20.07419    5.81125  -3.454 0.000552 ***
    ## avg_friend_age                1.56541    0.51838   3.020 0.002529 ** 
    ## avg_friend_male              -0.02011    0.10580  -0.190 0.849214    
    ## friend_country_cnt            4.74217    0.80194   5.913 3.35e-09 ***
    ## subscriber_friend_cnt         8.73413    3.29036   2.654 0.007944 ** 
    ## songsListened                 4.25557    0.84156   5.057 4.26e-07 ***
    ## lovedTracks                   5.21342    0.80796   6.453 1.10e-10 ***
    ## posts                        -0.02834    1.97451  -0.014 0.988549    
    ## playlists                     7.54911    1.77707   4.248 2.16e-05 ***
    ## shouts                        2.70373    2.23999   1.207 0.227422    
    ## delta_friend_cnt             -1.39030    3.22098  -0.432 0.666003    
    ## delta_avg_friend_age         -0.81645    1.93010  -0.423 0.672290    
    ## delta_avg_friend_male        -2.97045    1.04245  -2.849 0.004379 ** 
    ## delta_friend_country_cnt      3.76800    2.36161   1.596 0.110595    
    ## delta_subscriber_friend_cnt  -2.22131    1.54330  -1.439 0.150058    
    ## delta_songsListened           8.59673    2.72250   3.158 0.001590 ** 
    ## delta_lovedTracks            -2.38273    2.00987  -1.186 0.235814    
    ## delta_posts                   0.93835    1.56704   0.599 0.549301    
    ## delta_playlists               0.23569    1.74807   0.135 0.892747    
    ## delta_shouts                  7.97937    3.52472   2.264 0.023585 *  
    ## tenure                        0.14967    0.18672   0.802 0.422785    
    ## good_country                 -0.53526    0.07058  -7.584 3.35e-14 ***
    ## delta_good_country           -1.31411    1.64016  -0.801 0.423011    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 9877.8  on 31154  degrees of freedom
    ## Residual deviance: 9243.8  on 31129  degrees of freedom
    ## AIC: 9295.8
    ## 
    ## Number of Fisher Scoring iterations: 6

Prediction

``` r
pred_fit_log <- predict(fit_log, newdata = xyzdata_normalized_drop_user_id_test, type = "response")
```

Check the accuracy by measuring AUC

``` r
roc_log <- roc.curve(xyzdata_normalized_drop_user_id_test$adopter, pred_fit_log, col = "blue", lwd = 2)
```

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

``` r
roc_log$auc
```

    ## [1] 0.7101434

Generate a dataframe of cutoff and corresponding recall_p and precision.

``` r
# initialize the dataframe 
dashboard <- data.frame()

# initialize vectors
cutoff <- c()
precision <- c()
recall_p <- c()

# for loop to get corresponding recall_p and precision for each cutoff value
threshold <- roc_log$thresholds
for (i in 1:(length(threshold))){
  cutoff <- c(cutoff, threshold[i])
  binary_predictions <- ifelse(pred_fit_log >= threshold[i], 1, 0)
  confusion_matrix <- confusionMatrix(data = factor(binary_predictions), reference = xyzdata_normalized_drop_user_id_test$adopter, mode = "prec_recall", positive = "1")
  recall_p <- c(recall_p, roc_log$true.positive.rate[i])
  precision <- c(precision, confusion_matrix$byClass[["Precision"]]) 
}
```

    ## Warning in confusionMatrix.default(data = factor(binary_predictions), reference
    ## = xyzdata_normalized_drop_user_id_test$adopter, : Levels are not in the same
    ## order for reference and data. Refactoring data to match.

    ## Warning in confusionMatrix.default(data = factor(binary_predictions), reference
    ## = xyzdata_normalized_drop_user_id_test$adopter, : Levels are not in the same
    ## order for reference and data. Refactoring data to match.

``` r
dashboard <- data.frame(cutoff, recall_p, precision)
dashboard
```

    ##          cutoff    recall_p  precision
    ## 1          -Inf 1.000000000 0.03707270
    ## 2   0.005411072 1.000000000 0.03708341
    ## 3   0.011164597 0.997402597 0.03755134
    ## 4   0.011979637 0.994805195 0.03781596
    ## 5   0.012659513 0.992207792 0.03816184
    ## 6   0.013320419 0.989610390 0.03842663
    ## 7   0.013989324 0.989610390 0.03881023
    ## 8   0.014594294 0.987012987 0.03913894
    ## 9   0.015133852 0.981818182 0.03939552
    ## 10  0.015660246 0.976623377 0.03963736
    ## 11  0.016141592 0.974025974 0.03994461
    ## 12  0.016549920 0.974025974 0.04040078
    ## 13  0.016915659 0.971428571 0.04074074
    ## 14  0.017238250 0.963636364 0.04088605
    ## 15  0.017567184 0.961038961 0.04123482
    ## 16  0.017892679 0.958441558 0.04164316
    ## 17  0.018200336 0.955844156 0.04203793
    ## 18  0.018511074 0.955844156 0.04255811
    ## 19  0.018838130 0.950649351 0.04284209
    ## 20  0.019194196 0.948051948 0.04323620
    ## 21  0.019522923 0.945454545 0.04365555
    ## 22  0.019853663 0.942857143 0.04413911
    ## 23  0.020174178 0.937662338 0.04442530
    ## 24  0.020461161 0.937662338 0.04497322
    ## 25  0.020749370 0.929870130 0.04521915
    ## 26  0.021059664 0.927272727 0.04568723
    ## 27  0.021347352 0.922077922 0.04601426
    ## 28  0.021626483 0.919480519 0.04656669
    ## 29  0.021939152 0.909090909 0.04659832
    ## 30  0.022255766 0.906493506 0.04719405
    ## 31  0.022591206 0.898701299 0.04747530
    ## 32  0.022909097 0.896103896 0.04798998
    ## 33  0.023219942 0.893506494 0.04856699
    ## 34  0.023540465 0.890909091 0.04916858
    ## 35  0.023857995 0.883116883 0.04950495
    ## 36  0.024174750 0.877922078 0.04997043
    ## 37  0.024488682 0.875324675 0.05057023
    ## 38  0.024812366 0.872727273 0.05120390
    ## 39  0.025146554 0.867532468 0.05177492
    ## 40  0.025477477 0.864935065 0.05248227
    ## 41  0.025822755 0.857142857 0.05287614
    ## 42  0.026181655 0.851948052 0.05342890
    ## 43  0.026530773 0.849350649 0.05419291
    ## 44  0.026875602 0.844155844 0.05475067
    ## 45  0.027231488 0.836363636 0.05526948
    ## 46  0.027581416 0.831168831 0.05591473
    ## 47  0.027890648 0.831168831 0.05699020
    ## 48  0.028220836 0.825974026 0.05776567
    ## 49  0.028567623 0.818181818 0.05834414
    ## 50  0.028901365 0.812987013 0.05904546
    ## 51  0.029251428 0.800000000 0.05932203
    ## 52  0.029616991 0.789610390 0.05984252
    ## 53  0.029979370 0.779220779 0.06022887
    ## 54  0.030344034 0.763636364 0.06030769
    ## 55  0.030715529 0.763636364 0.06167401
    ## 56  0.031097983 0.758441558 0.06250000
    ## 57  0.031473089 0.750649351 0.06333552
    ## 58  0.031843578 0.735064935 0.06339606
    ## 59  0.032236826 0.732467532 0.06485741
    ## 60  0.032654746 0.724675325 0.06563162
    ## 61  0.033085775 0.722077922 0.06718221
    ## 62  0.033520997 0.711688312 0.06793950
    ## 63  0.033962283 0.701298701 0.06873727
    ## 64  0.034436695 0.683116883 0.06870428
    ## 65  0.034940583 0.680519481 0.07035446
    ## 66  0.035400382 0.662337662 0.07032543
    ## 67  0.035850143 0.651948052 0.07150997
    ## 68  0.036327009 0.641558442 0.07239156
    ## 69  0.036828497 0.625974026 0.07307459
    ## 70  0.037383899 0.602597403 0.07279573
    ## 71  0.037983242 0.589610390 0.07372524
    ## 72  0.038621464 0.574025974 0.07391304
    ## 73  0.039225920 0.566233766 0.07569444
    ## 74  0.039807825 0.550649351 0.07623157
    ## 75  0.040493260 0.537662338 0.07738318
    ## 76  0.041236432 0.524675325 0.07853810
    ## 77  0.041952632 0.496103896 0.07770545
    ## 78  0.042664246 0.477922078 0.07799915
    ## 79  0.043438875 0.475324675 0.08126110
    ## 80  0.044291518 0.462337662 0.08282922
    ## 81  0.045290950 0.428571429 0.08044856
    ## 82  0.046345215 0.418181818 0.08273381
    ## 83  0.047457522 0.402597403 0.08401084
    ## 84  0.048647014 0.394805195 0.08775982
    ## 85  0.049823063 0.376623377 0.08917589
    ## 86  0.051140066 0.340259740 0.08635465
    ## 87  0.052547642 0.322077922 0.08825623
    ## 88  0.054054005 0.309090909 0.09049430
    ## 89  0.055714259 0.288311688 0.09226933
    ## 90  0.057725022 0.264935065 0.09383625
    ## 91  0.060323052 0.259740260 0.10010010
    ## 92  0.063175319 0.236363636 0.10167598
    ## 93  0.066279958 0.212987013 0.10485934
    ## 94  0.070041114 0.189610390 0.10782866
    ## 95  0.074509112 0.166233766 0.11091854
    ## 96  0.081018735 0.137662338 0.11572052
    ## 97  0.090700510 0.114285714 0.11796247
    ## 98  0.105815910 0.067532468 0.10569106
    ## 99  0.135717630 0.051948052 0.13605442
    ## 100 0.569871403 0.007792208 0.33333333
    ## 101         Inf 0.000000000         NA

### Cross validation for logistics regression, oversampling, no feature selection

We use 10 folds cross-validation. Note that if we want to combind
cross-validation and oversampling, we should oversample the 9 folds as
training each time INSIDE the loop.

``` r
library(caret)

# create a list of row indexes that correspond to each folds
cv <- createFolds(y = xyzdata_normalized_drop_user_id$adopter, k = 10) 

# a vector to store auc from each fold
AUC_cv <- c()

for(test_rows in cv){
  xyz_train <- xyzdata_normalized_drop_user_id[-test_rows, ]
  xyz_test <- xyzdata_normalized_drop_user_id[test_rows, ]
  
  # oversample the training set
  library(ROSE)
  xyz_train_oversample_cv <- ROSE(adopter ~., data = xyz_train, seed = 123)$data
  
  # train the model then evaluate its performance
  fit_log_cv <- glm(adopter ~ ., data = xyz_train_oversample_cv, family = binomial)
  
  # predict
  pred_fit_log_cv <- predict(fit_log_cv, newdata = xyz_test, type = "response")
  
  # get auc
  roc_cv <- roc(xyz_test$adopter, pred_fit_log_cv, col = "blue", lwd = 2)
  auc_cv <- roc_cv$auc
  
  # add auc of current folds
  AUC_cv <- c(AUC_cv, auc_cv)
}
```

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

``` r
# report average accuracy across folds
mean(AUC_cv)
```

    ## [1] 0.7489558

Oversampling training set helps in performance

Check multicollinearity

``` r
library(car)
```

    ## Loading required package: carData

    ## 
    ## Attaching package: 'car'

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     recode

``` r
vif(fit_log_cv)
```

    ##                         age                        male 
    ##                    1.314098                    1.025190 
    ##                  friend_cnt              avg_friend_age 
    ##                    1.249086                    1.323349 
    ##             avg_friend_male          friend_country_cnt 
    ##                    1.016955                    1.292105 
    ##       subscriber_friend_cnt               songsListened 
    ##                    1.125246                    1.171440 
    ##                 lovedTracks                       posts 
    ##                    1.065334                    1.065651 
    ##                   playlists                      shouts 
    ##                    1.024567                    1.104978 
    ##            delta_friend_cnt        delta_avg_friend_age 
    ##                    1.157505                    1.029067 
    ##       delta_avg_friend_male    delta_friend_country_cnt 
    ##                    1.027868                    1.117495 
    ## delta_subscriber_friend_cnt         delta_songsListened 
    ##                    1.025287                    1.115015 
    ##           delta_lovedTracks                 delta_posts 
    ##                    1.066572                    1.039236 
    ##             delta_playlists                delta_shouts 
    ##                    1.016743                    1.040715 
    ##                      tenure                good_country 
    ##                    1.115806                    1.016345 
    ##          delta_good_country 
    ##                    1.005789

Variance Inflation Factors: VIF = 1/(1 - R_squared^2), detects
multicollinearity in regression analysis. Multicollinearity happens when
independent variables in a regression model are highly correlated to
each other, making it hard to interpret the model and also causes
problems in performance.

Reading VIF: VIF of 1.9 indicates the variance of a particular
coefficient is 90% higher than what we would expect if there was no
multicollinearity, i.e. the variance of a particular coefficient is 90%
higher than being orthogonal.

Usually VIF \< 2 is not going to cause problems, which is the case here.
But we still want to do PCA later since too many variables makes the
model hard to interpret and some variables might measure similar
factors.

## Feature Selection: Filter Approach

### Filtering

``` r
library(FSelectorRcpp)
IG <- information_gain(adopter ~ ., data = xyzdata_normalized_drop_user_id)

# select top 10
top10 <- cut_attrs(IG, k = 10)

# the whole normalized dataset
xyzdata_normalized_drop_user_id_top10 <- xyzdata_normalized_drop_user_id %>% select(top10, adopter)
```

    ## Warning: Using an external vector in selections was deprecated in tidyselect 1.1.0.
    ## ℹ Please use `all_of()` or `any_of()` instead.
    ##   # Was:
    ##   data %>% select(top10)
    ## 
    ##   # Now:
    ##   data %>% select(all_of(top10))
    ## 
    ## See <https://tidyselect.r-lib.org/reference/faq-external-vector.html>.
    ## This warning is displayed once every 8 hours.
    ## Call `lifecycle::last_lifecycle_warnings()` to see where this warning was
    ## generated.

``` r
# training set
xyzdata_normalized_drop_user_id_train_top10 <- xyzdata_normalized_drop_user_id_train %>% select(top10, adopter)

# testing set
xyzdata_normalized_drop_user_id_test_top10 <- xyzdata_normalized_drop_user_id_test %>% select(top10, adopter)
```

lovedTracks + delta_songsListened + delta_lovedTracks +
subscriber_friend_cnt + songsListened + friend_cnt +
friend_country_cnt + delta_friend_cnt + delta_subscriber_friend_cnt +
delta_avg_friend_male

### Cross validation for logistics regression, oversampling, filtered

We use 10 folds cross-validation

``` r
library(caret)

# create a list of row indexes that correspond to each folds
cv <- createFolds(y = xyzdata_normalized_drop_user_id_top10$adopter, k = 10)

# a vector to store auc from each fold
AUC_cv_filter <- c()

for(test_rows in cv){
  xyz_train_f <- xyzdata_normalized_drop_user_id_top10[-test_rows, ]
  xyz_test_f <- xyzdata_normalized_drop_user_id_top10[test_rows, ]
  
  # oversample the training folds
  xyz_train_oversample_filter <- ROSE(adopter ~ lovedTracks + delta_songsListened + delta_lovedTracks + subscriber_friend_cnt + songsListened + friend_cnt + friend_country_cnt + delta_friend_cnt + delta_subscriber_friend_cnt + delta_avg_friend_male, data = xyz_train_f, seed = 123)$data
  
  # train the model then evaluate its performance
  fit_log_oversample_filter_cv <- glm(adopter ~ lovedTracks + delta_songsListened + delta_lovedTracks + subscriber_friend_cnt + songsListened + friend_cnt + friend_country_cnt + delta_friend_cnt + delta_subscriber_friend_cnt + delta_avg_friend_male, data = xyz_train_oversample_filter, family = binomial)
  
  # predict
  pred_fit_log_ftiler_cv <- predict(fit_log_oversample_filter_cv, newdata = xyz_test_f, type = "response")
  
  # get auc
  roc_cv_filter <- roc.curve(xyz_test_f$adopter, pred_fit_log_ftiler_cv)
  auc_cv_filter <- roc_cv_filter$auc
  
  # add auc of current folds
  AUC_cv_filter <- c(AUC_cv_filter, auc_cv_filter)
}
```

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-14-2.png)<!-- -->![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-14-3.png)<!-- -->

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-14-4.png)<!-- -->![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-14-5.png)<!-- -->![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-14-6.png)<!-- -->![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-14-7.png)<!-- -->![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-14-8.png)<!-- -->![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-14-9.png)<!-- -->![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-14-10.png)<!-- -->

``` r
# report average accuracy across folds
mean(AUC_cv_filter) 
```

    ## [1] 0.7413279

Check multicollinearity

``` r
vif(fit_log_oversample_filter_cv)
```

    ##                 lovedTracks         delta_songsListened 
    ##                    1.083114                    1.151555 
    ##           delta_lovedTracks       subscriber_friend_cnt 
    ##                    1.090058                    1.196114 
    ##               songsListened                  friend_cnt 
    ##                    1.169008                    1.407244 
    ##          friend_country_cnt            delta_friend_cnt 
    ##                    1.444149                    1.115594 
    ## delta_subscriber_friend_cnt       delta_avg_friend_male 
    ##                    1.021396                    1.001556

So far, we’ve seen some important points: 1. oversampling is needed. 2.
no severe multicollinearity problems needed to concern, but we will
still try PCA later. 3. filtered as top 10 variables act in relatively
clear pattern in EDA previously.

Generate a dataframe of cutoff and corresponding recall_p and precision.

``` r
# initialize the dataframe 
dashboard_filter_log <- data.frame()

# initialize vectors
cutoff_filter_log <- c()
precision_filter_log <- c()
recall_p_filter_log <- c()

# for loop to get corresponding recall_p and precision for each cutoff value
threshold <- roc_cv_filter$thresholds
for (i in 1:(length(threshold))){
  cutoff_filter_log <- c(cutoff_filter_log, threshold[i])
  binary_predictions <- ifelse(pred_fit_log_ftiler_cv >= threshold[i], 1, 0)
  confusion_matrix <- confusionMatrix(data = factor(binary_predictions), reference = xyz_test_f$adopter, mode = "prec_recall", positive = "1")
  recall_p_filter_log <- c(recall_p_filter_log, roc_cv_filter$true.positive.rate[i])
  precision_filter_log <- c(precision_filter_log, confusion_matrix$byClass[["Precision"]]) 
}
```

    ## Warning in confusionMatrix.default(data = factor(binary_predictions), reference
    ## = xyz_test_f$adopter, : Levels are not in the same order for reference and
    ## data. Refactoring data to match.

    ## Warning in confusionMatrix.default(data = factor(binary_predictions), reference
    ## = xyz_test_f$adopter, : Levels are not in the same order for reference and
    ## data. Refactoring data to match.

``` r
dashboard_filter_log <- data.frame(cutoff_filter_log, recall_p_filter_log, precision_filter_log)
dashboard_filter_log
```

    ##     cutoff_filter_log recall_p_filter_log precision_filter_log
    ## 1                -Inf         1.000000000           0.03707270
    ## 2           0.2553300         1.000000000           0.03709949
    ## 3           0.3665202         0.993506494           0.03726254
    ## 4           0.3715558         0.993506494           0.03771260
    ## 5           0.3718179         0.987012987           0.03905447
    ## 6           0.3719519         0.987012987           0.03942931
    ## 7           0.3721398         0.987012987           0.03984273
    ## 8           0.3723844         0.980519481           0.04003181
    ## 9           0.3726462         0.980519481           0.04042838
    ## 10          0.3729173         0.974025974           0.04061738
    ## 11          0.3732752         0.974025974           0.04110715
    ## 12          0.3736990         0.967532468           0.04127424
    ## 13          0.3741634         0.967532468           0.04184218
    ## 14          0.3747084         0.967532468           0.04223356
    ## 15          0.3753262         0.967532468           0.04282840
    ## 16          0.3759284         0.967532468           0.04326365
    ## 17          0.3763906         0.967532468           0.04375918
    ## 18          0.3768006         0.967532468           0.04442457
    ## 19          0.3772591         0.967532468           0.04489304
    ## 20          0.3777707         0.967532468           0.04555182
    ## 21          0.3782850         0.967532468           0.04608723
    ## 22          0.3788205         0.961038961           0.04632238
    ## 23          0.3794952         0.961038961           0.04689480
    ## 24          0.3801802         0.961038961           0.04752730
    ## 25          0.3808212         0.961038961           0.04806755
    ## 26          0.3814993         0.961038961           0.04878049
    ## 27          0.3822065         0.961038961           0.04949833
    ## 28          0.3829804         0.961038961           0.05008460
    ## 29          0.3838857         0.961038961           0.05082418
    ## 30          0.3849129         0.954545455           0.05125523
    ## 31          0.3859416         0.954545455           0.05199859
    ## 32          0.3869075         0.954545455           0.05266929
    ## 33          0.3877884         0.948051948           0.05309091
    ## 34          0.3887440         0.928571429           0.05290418
    ## 35          0.3900031         0.928571429           0.05347794
    ## 36          0.3914230         0.915584416           0.05369383
    ## 37          0.3927386         0.915584416           0.05456656
    ## 38          0.3939525         0.915584416           0.05544632
    ## 39          0.3951780         0.915584416           0.05637745
    ## 40          0.3965720         0.902597403           0.05648111
    ## 41          0.3978985         0.902597403           0.05748553
    ## 42          0.3991680         0.896103896           0.05788591
    ## 43          0.4006415         0.889610390           0.05854701
    ## 44          0.4021989         0.889610390           0.05959113
    ## 45          0.4036303         0.863636364           0.05871965
    ## 46          0.4050166         0.831168831           0.05765766
    ## 47          0.4066215         0.811688312           0.05741847
    ## 48          0.4083132         0.811688312           0.05835668
    ## 49          0.4100430         0.798701299           0.05862726
    ## 50          0.4114741         0.792207792           0.05933852
    ## 51          0.4128420         0.785714286           0.06004963
    ## 52          0.4145845         0.785714286           0.06139016
    ## 53          0.4164126         0.772727273           0.06153051
    ## 54          0.4183327         0.753246753           0.06114918
    ## 55          0.4203142         0.720779221           0.05996759
    ## 56          0.4222516         0.707792208           0.06042129
    ## 57          0.4240354         0.707792208           0.06137387
    ## 58          0.4260878         0.707792208           0.06289671
    ## 59          0.4284154         0.681818182           0.06216696
    ## 60          0.4308158         0.649350649           0.06060606
    ## 61          0.4331357         0.642857143           0.06168224
    ## 62          0.4351628         0.636363636           0.06242038
    ## 63          0.4374856         0.629870130           0.06369009
    ## 64          0.4400788         0.623376623           0.06451613
    ## 65          0.4430116         0.610389610           0.06523248
    ## 66          0.4459552         0.603896104           0.06600426
    ## 67          0.4487214         0.590909091           0.06656913
    ## 68          0.4519922         0.577922078           0.06747536
    ## 69          0.4549241         0.571428571           0.06885759
    ## 70          0.4576215         0.564935065           0.07016129
    ## 71          0.4608823         0.558441558           0.07136929
    ## 72          0.4647877         0.538961039           0.07155172
    ## 73          0.4690014         0.538961039           0.07397504
    ## 74          0.4731872         0.525974026           0.07555970
    ## 75          0.4772120         0.519480519           0.07699711
    ## 76          0.4809774         0.512987013           0.07900000
    ## 77          0.4855836         0.512987013           0.08246347
    ## 78          0.4912550         0.500000000           0.08415301
    ## 79          0.4963844         0.493506494           0.08685714
    ## 80          0.5012927         0.474025974           0.08711217
    ## 81          0.5065899         0.461038961           0.08919598
    ## 82          0.5110277         0.454545455           0.09333333
    ## 83          0.5153706         0.441558442           0.09577465
    ## 84          0.5205059         0.441558442           0.10104012
    ## 85          0.5275486         0.409090909           0.09952607
    ## 86          0.5361984         0.383116883           0.10000000
    ## 87          0.5469354         0.350649351           0.09747292
    ## 88          0.5575843         0.331168831           0.10079051
    ## 89          0.5675452         0.318181818           0.10425532
    ## 90          0.5788846         0.285714286           0.10256410
    ## 91          0.5927726         0.272727273           0.10937500
    ## 92          0.6072127         0.246753247           0.10982659
    ## 93          0.6221135         0.227272727           0.11627907
    ## 94          0.6392739         0.188311688           0.11196911
    ## 95          0.6572604         0.162337662           0.11363636
    ## 96          0.6799630         0.129870130           0.10989011
    ## 97          0.7158846         0.110389610           0.12056738
    ## 98          0.7597021         0.077922078           0.11428571
    ## 99          0.8214377         0.032467532           0.08771930
    ## 100         0.9310767         0.006493506           0.04761905
    ## 101               Inf         0.000000000                   NA

## PCA: oversampling, logistic regression, only transform relatively highly correlated variables

``` r
# we use oversampled training set
library(ROSE)
xyz_train_oversample_pca <- ROSE(adopter ~., data = xyzdata_normalized_drop_user_id_train, seed = 123)$data

# get PCs
high_cor <- xyz_train_oversample_pca[, c(1, 3, 4, 6, 7, 8, 9, 12, 13, 16, 18, 19, 22)]
xyzdata_rose_eigen_high_cor <- eigen(cor(high_cor))
xyzdata_rose_pca_high_cor <- prcomp(high_cor)
summary(xyzdata_rose_pca_high_cor)
```

    ## Importance of components:
    ##                           PC1     PC2     PC3     PC4     PC5     PC6     PC7
    ## Standard deviation     0.1282 0.08343 0.07152 0.04788 0.04405 0.02706 0.01916
    ## Proportion of Variance 0.4689 0.19873 0.14606 0.06545 0.05540 0.02091 0.01048
    ## Cumulative Proportion  0.4689 0.66767 0.81373 0.87918 0.93458 0.95549 0.96598
    ##                            PC8     PC9    PC10    PC11    PC12    PC13
    ## Standard deviation     0.01766 0.01623 0.01386 0.01275 0.01230 0.01051
    ## Proportion of Variance 0.00890 0.00752 0.00548 0.00464 0.00432 0.00316
    ## Cumulative Proportion  0.97488 0.98240 0.98788 0.99252 0.99684 1.00000

``` r
# colnames(xyz_train_oversample_pca[, c(1, 3, 4, 6, 7, 8, 9, 12, 13, 16, 18, 19, 22)])

screeplot(xyzdata_rose_pca_high_cor, type = "lines")
```

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->
From the screeplot and summary of PCs, it seems the first 6 PCs are more
important since the cumulative proportion of variance explained by them
is about 95.9% for those variables with higher correlations.

We also check relationships of PCs and those variables with higher
correlations.

``` r
xyzdata_rose_pca_high_cor$rotation
```

    ##                                   PC1         PC2           PC3          PC4
    ## age                       0.847806523 -0.02223507  0.5294298471 -0.014271479
    ## friend_cnt               -0.008885638 -0.09323827  0.0038856245 -0.000472559
    ## avg_friend_age            0.528207591 -0.03496533 -0.8475816186 -0.013754011
    ## friend_country_cnt       -0.038381557 -0.93417636  0.0153551617 -0.302368302
    ## subscriber_friend_cnt     0.004308842 -0.08449343 -0.0042947759  0.008033588
    ## songsListened            -0.005686050 -0.20187484  0.0277057304  0.432130848
    ## lovedTracks               0.013714307 -0.22089310 -0.0141432946  0.839749088
    ## shouts                   -0.008405937 -0.07967952  0.0022466442  0.023061339
    ## delta_friend_cnt         -0.011599745 -0.07091252  0.0042130620  0.004817776
    ## delta_friend_country_cnt -0.008394891 -0.05625122  0.0034737920 -0.004128563
    ## delta_songsListened      -0.006218026 -0.03946790  0.0053268698  0.054707528
    ## delta_lovedTracks        -0.006201057 -0.04782459  0.0009186227  0.111939136
    ## delta_shouts             -0.008772583 -0.04279379  0.0028962568  0.010238880
    ##                                   PC5          PC6           PC7          PC8
    ## age                      -0.008864010  0.010114543  0.0001228203  0.002938856
    ## friend_cnt                0.002995315  0.068464742 -0.0243892550  0.210908878
    ## avg_friend_age            0.029690274  0.013153655  0.0025834983  0.001115965
    ## friend_country_cnt       -0.074350053 -0.114648910  0.0052096955 -0.088821272
    ## subscriber_friend_cnt    -0.008906411  0.052954260 -0.0435125703  0.207223584
    ## songsListened             0.869552530 -0.007566828  0.0147902852 -0.044373336
    ## lovedTracks              -0.470397689 -0.082595470 -0.0110919806 -0.028689221
    ## shouts                    0.018871311  0.205767568 -0.5549921246  0.742529561
    ## delta_friend_cnt         -0.029138771  0.431233366  0.2293105267  0.107296189
    ## delta_friend_country_cnt -0.019451636  0.462207602  0.6313790516  0.259228911
    ## delta_songsListened       0.092729198  0.089940942  0.1176960326  0.021420074
    ## delta_lovedTracks        -0.071949596  0.186824772  0.1546905369  0.022234312
    ## delta_shouts             -0.025866829  0.698053163 -0.4473428264 -0.520450989
    ##                                   PC9          PC10         PC11          PC12
    ## age                      -0.003504340 -9.777508e-05  0.004014058  0.0002952097
    ## friend_cnt               -0.024514609  5.272434e-01 -0.160972789 -0.0908808611
    ## avg_friend_age           -0.004382358 -1.653087e-03  0.007314735  0.0014382647
    ## friend_country_cnt       -0.002881608 -7.828004e-02  0.029837769  0.0169807368
    ## subscriber_friend_cnt     0.043768784  5.176689e-01 -0.534947988 -0.2845924574
    ## songsListened             0.011597849 -1.584902e-02 -0.058072718  0.0978307542
    ## lovedTracks               0.125501664 -1.586897e-02  0.022314903 -0.0007910834
    ## shouts                   -0.019448322 -2.726536e-01  0.119498752  0.0292560373
    ## delta_friend_cnt          0.054883361  4.612741e-01  0.492187349  0.4890091032
    ## delta_friend_country_cnt  0.269013128 -3.801997e-01 -0.275724383 -0.1220708382
    ## delta_songsListened      -0.102320982  8.344867e-02  0.560446412 -0.7938969831
    ## delta_lovedTracks        -0.943391561 -8.046535e-02 -0.141673215  0.0608429515
    ## delta_shouts              0.073093472 -4.791590e-02 -0.127923222 -0.1096559441
    ##                                  PC13
    ## age                       0.003470938
    ## friend_cnt                0.792893703
    ## avg_friend_age            0.007101481
    ## friend_country_cnt       -0.016602389
    ## subscriber_friend_cnt    -0.554961158
    ## songsListened            -0.003733260
    ## lovedTracks               0.009819402
    ## shouts                   -0.033567033
    ## delta_friend_cnt         -0.216158690
    ## delta_friend_country_cnt  0.095067224
    ## delta_songsListened      -0.050766270
    ## delta_lovedTracks        -0.020097276
    ## delta_shouts              0.054945397

(will be replaced with the labeled picture PDF file after rendering.)

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-19-2.png)<!-- -->![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-19-3.png)<!-- -->![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-19-4.png)<!-- -->![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-19-5.png)<!-- -->![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-19-6.png)<!-- -->
Also consider the plots above: Until now, we decide to extract the first
6 PCs except for the 3rd since the variable it measures is already in
PC1 and its magnitude is too small and important information may already
be included in PC1. PC1: capture age_related information, maybe usage
and age, including age, avg_friend_age. PC2: capture friend_diversity
information, including friend_country_cnt. PC4: capture
tend_to_exploration information, meaning that users might have many
already loved content and also tend to try new things, including
songsListened, lovedTracks PC5: capture tend_to_concentration
information, meaning that users might have many already loved content
but not tend to try new things, including songsListened, lovedTracks
PC6: capture information_received ability since delta_shots is the
changed number of wall posts received, and increase in number of friends
and friend’s countries will also have effects on information_received
ability, including delta_shouts, delta_friend_cnt,
delta_friend_country_cnt.

Thus we identify the 5 PCs: PC1(age_related), PC2(friend_diversity),
PC4(tend_to_exploration), PC5(tend_to_concentration),
PC6(information_received).

Prepare PCA training and testing set for PCA model.

``` r
# remained
remained_train <- xyz_train_oversample_pca[, c(-1, -3, -4, -6, -7, -8, -9, -12, -13, -16, -18, -19, -22)]
remained_test <- xyzdata_normalized_drop_user_id_test[, c(-1, -3, -4, -6, -7, -8, -9, -12, -13, -16, -18, -19, -22)]

# prepare PCA training set
train_data_pca <- predict(xyzdata_rose_pca_high_cor, xyz_train_oversample_pca[, -which(names(xyzdata_normalized_drop_user_id_test) == "adopter")])
train_data_pca_6pc <- train_data_pca[, c(1:2, 4:6)] #give up pc3
train_data_pca_6pcAndremained <- cbind(remained_train, train_data_pca_6pc)

# prepare PCA testing set
test_data_pca <- predict(xyzdata_rose_pca_high_cor, xyzdata_normalized_drop_user_id_test[, -which(names(xyzdata_normalized_drop_user_id_test) == "adopter")])
test_data_pca_6pc <- test_data_pca[, c(1:2, 4:6)] #give up pc3
test_data_pca_6pcAndremained <- cbind(remained_test, test_data_pca_6pc)

# fit the model
fit_PCA <- glm(adopter ~ PC1 + PC2 + PC4 + PC5 + PC6 + male + avg_friend_male + posts + playlists + delta_avg_friend_age + delta_avg_friend_male + delta_subscriber_friend_cnt + delta_posts + delta_playlists + tenure + good_country + delta_good_country, data = train_data_pca_6pcAndremained, family = binomial)

# check variance inflation factors
library(car)
vif(fit_log_cv) # logistics regression, oversampling, no feature selection
```

    ##                         age                        male 
    ##                    1.314098                    1.025190 
    ##                  friend_cnt              avg_friend_age 
    ##                    1.249086                    1.323349 
    ##             avg_friend_male          friend_country_cnt 
    ##                    1.016955                    1.292105 
    ##       subscriber_friend_cnt               songsListened 
    ##                    1.125246                    1.171440 
    ##                 lovedTracks                       posts 
    ##                    1.065334                    1.065651 
    ##                   playlists                      shouts 
    ##                    1.024567                    1.104978 
    ##            delta_friend_cnt        delta_avg_friend_age 
    ##                    1.157505                    1.029067 
    ##       delta_avg_friend_male    delta_friend_country_cnt 
    ##                    1.027868                    1.117495 
    ## delta_subscriber_friend_cnt         delta_songsListened 
    ##                    1.025287                    1.115015 
    ##           delta_lovedTracks                 delta_posts 
    ##                    1.066572                    1.039236 
    ##             delta_playlists                delta_shouts 
    ##                    1.016743                    1.040715 
    ##                      tenure                good_country 
    ##                    1.115806                    1.016345 
    ##          delta_good_country 
    ##                    1.005789

``` r
vif(fit_PCA)
```

    ##                         PC1                         PC2 
    ##                    1.074669                    1.108672 
    ##                         PC4                         PC5 
    ##                    1.052851                    1.053206 
    ##                         PC6                        male 
    ##                    1.076266                    1.011950 
    ##             avg_friend_male                       posts 
    ##                    1.006763                    1.050491 
    ##                   playlists        delta_avg_friend_age 
    ##                    1.025228                    1.023225 
    ##       delta_avg_friend_male delta_subscriber_friend_cnt 
    ##                    1.024827                    1.013347 
    ##                 delta_posts             delta_playlists 
    ##                    1.042790                    1.021323 
    ##                      tenure                good_country 
    ##                    1.099304                    1.016207 
    ##          delta_good_country 
    ##                    1.006916

Check VIFs of the logistics regression, oversampling, no feature
selection again, they are not in a big problem of multicollinearity, but
after fitting PCs in the model, they become lower, which is a good
thing.

And if we check again summary of (logistics regression, oversampling, no
feature selection) and (logistic regression, oversampling, PCA for
transforming relatively highly correlated variables):

``` r
summary(fit_log)
```

    ## 
    ## Call:
    ## glm(formula = adopter ~ ., family = binomial, data = xyzdata_normalized_drop_user_id_train)
    ## 
    ## Coefficients:
    ##                              Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)                  -6.20997    1.92362  -3.228 0.001245 ** 
    ## age                           1.64869    0.40415   4.079 4.52e-05 ***
    ## male                          0.38669    0.07024   5.505 3.68e-08 ***
    ## friend_cnt                  -20.07419    5.81125  -3.454 0.000552 ***
    ## avg_friend_age                1.56541    0.51838   3.020 0.002529 ** 
    ## avg_friend_male              -0.02011    0.10580  -0.190 0.849214    
    ## friend_country_cnt            4.74217    0.80194   5.913 3.35e-09 ***
    ## subscriber_friend_cnt         8.73413    3.29036   2.654 0.007944 ** 
    ## songsListened                 4.25557    0.84156   5.057 4.26e-07 ***
    ## lovedTracks                   5.21342    0.80796   6.453 1.10e-10 ***
    ## posts                        -0.02834    1.97451  -0.014 0.988549    
    ## playlists                     7.54911    1.77707   4.248 2.16e-05 ***
    ## shouts                        2.70373    2.23999   1.207 0.227422    
    ## delta_friend_cnt             -1.39030    3.22098  -0.432 0.666003    
    ## delta_avg_friend_age         -0.81645    1.93010  -0.423 0.672290    
    ## delta_avg_friend_male        -2.97045    1.04245  -2.849 0.004379 ** 
    ## delta_friend_country_cnt      3.76800    2.36161   1.596 0.110595    
    ## delta_subscriber_friend_cnt  -2.22131    1.54330  -1.439 0.150058    
    ## delta_songsListened           8.59673    2.72250   3.158 0.001590 ** 
    ## delta_lovedTracks            -2.38273    2.00987  -1.186 0.235814    
    ## delta_posts                   0.93835    1.56704   0.599 0.549301    
    ## delta_playlists               0.23569    1.74807   0.135 0.892747    
    ## delta_shouts                  7.97937    3.52472   2.264 0.023585 *  
    ## tenure                        0.14967    0.18672   0.802 0.422785    
    ## good_country                 -0.53526    0.07058  -7.584 3.35e-14 ***
    ## delta_good_country           -1.31411    1.64016  -0.801 0.423011    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 9877.8  on 31154  degrees of freedom
    ## Residual deviance: 9243.8  on 31129  degrees of freedom
    ## AIC: 9295.8
    ## 
    ## Number of Fisher Scoring iterations: 6

``` r
summary(fit_PCA)
```

    ## 
    ## Call:
    ## glm(formula = adopter ~ PC1 + PC2 + PC4 + PC5 + PC6 + male + 
    ##     avg_friend_male + posts + playlists + delta_avg_friend_age + 
    ##     delta_avg_friend_male + delta_subscriber_friend_cnt + delta_posts + 
    ##     delta_playlists + tenure + good_country + delta_good_country, 
    ##     family = binomial, data = train_data_pca_6pcAndremained)
    ## 
    ## Coefficients:
    ##                             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)                  1.46684    0.45498   3.224  0.00126 ** 
    ## PC1                          2.04030    0.09984  20.437  < 2e-16 ***
    ## PC2                         -6.50495    0.18126 -35.887  < 2e-16 ***
    ## PC4                          7.18502    0.31050  23.140  < 2e-16 ***
    ## PC5                          2.02091    0.31662   6.383 1.74e-10 ***
    ## PC6                          6.70360    0.68479   9.789  < 2e-16 ***
    ## male                         0.27870    0.02133  13.068  < 2e-16 ***
    ## avg_friend_male             -0.02126    0.03471  -0.612  0.54030    
    ## posts                        1.99906    1.17700   1.698  0.08943 .  
    ## playlists                    5.54101    0.60328   9.185  < 2e-16 ***
    ## delta_avg_friend_age        -0.88764    0.61969  -1.432  0.15203    
    ## delta_avg_friend_male       -1.99522    0.32640  -6.113 9.79e-10 ***
    ## delta_subscriber_friend_cnt  0.38095    0.47404   0.804  0.42162    
    ## delta_posts                  1.25144    0.80066   1.563  0.11805    
    ## delta_playlists              0.53739    0.59958   0.896  0.37010    
    ## tenure                       0.06877    0.05529   1.244  0.21356    
    ## good_country                -0.34098    0.02152 -15.845  < 2e-16 ***
    ## delta_good_country          -1.24287    0.42598  -2.918  0.00353 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 43186  on 31154  degrees of freedom
    ## Residual deviance: 39760  on 31137  degrees of freedom
    ## AIC: 39796
    ## 
    ## Number of Fisher Scoring iterations: 4

There become less insignificant variables in the latter and we also
reduce dimension, making the model more easily to explain. So so far we
prefer PCA logistic regression model than logistic regression without
feature selection.

Prediction

``` r
# predict
pred_fit_PCA <- predict(fit_PCA, newdata = test_data_pca_6pcAndremained, type = "response")

# auc
roc_pca <- roc.curve(test_data_pca_6pcAndremained$adopter, pred_fit_PCA, , col = "blue", lwd = 2)
```

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-22-1.png)<!-- -->

``` r
auc_pca <- roc_pca$auc
roc_pca
```

    ## Area under the curve (AUC): 0.715

If AUC not changing a lot, a more easily explainable model is preferred.

Generate a dataframe of cutoff and corresponding recall_p and precision.

``` r
# initialize the dataframe 
dashboard_pca <- data.frame()

# initialize vectors
cutoff_pca <- c()
precision_pca <- c()
recall_p_pca <- c()

# for loop to get corresponding recall_p and precision for each cutoff value
threshold <- roc_pca$thresholds
for (i in 1:(length(threshold))){
  cutoff_pca <- c(cutoff_pca, threshold[i])
  binary_predictions <- ifelse(pred_fit_PCA >= threshold[i], 1, 0)
  confusion_matrix <- confusionMatrix(data = factor(binary_predictions), reference = test_data_pca_6pcAndremained$adopter, mode = "prec_recall", positive = "1")
  recall_p_pca <- c(recall_p_pca, roc_pca$true.positive.rate[i])
  precision_pca <- c(precision_pca, confusion_matrix$byClass[["Precision"]]) 
}
```

    ## Warning in confusionMatrix.default(data = factor(binary_predictions), reference
    ## = test_data_pca_6pcAndremained$adopter, : Levels are not in the same order for
    ## reference and data. Refactoring data to match.

    ## Warning in confusionMatrix.default(data = factor(binary_predictions), reference
    ## = test_data_pca_6pcAndremained$adopter, : Levels are not in the same order for
    ## reference and data. Refactoring data to match.

``` r
dashboard_pca <- data.frame(cutoff_pca, recall_p_pca, precision_pca)
dashboard_pca
```

    ##     cutoff_pca recall_p_pca precision_pca
    ## 1         -Inf   1.00000000    0.03707270
    ## 2    0.1801328   1.00000000    0.03708341
    ## 3    0.2535168   0.99740260    0.03756603
    ## 4    0.2661188   0.99480519    0.03784585
    ## 5    0.2754258   0.99480519    0.03820449
    ## 6    0.2827967   0.99220779    0.03852748
    ## 7    0.2897779   0.99220779    0.03893986
    ## 8    0.2963073   0.99220779    0.03936927
    ## 9    0.3021079   0.99220779    0.03982070
    ## 10   0.3068036   0.98961039    0.04014329
    ## 11   0.3107674   0.98961039    0.04059670
    ## 12   0.3149670   0.98961039    0.04106046
    ## 13   0.3189091   0.98441558    0.04126293
    ## 14   0.3223509   0.98441558    0.04177229
    ## 15   0.3255493   0.98441558    0.04225195
    ## 16   0.3284099   0.97402597    0.04227734
    ## 17   0.3315159   0.97402597    0.04279356
    ## 18   0.3348808   0.97142857    0.04324199
    ## 19   0.3380027   0.96883117    0.04361553
    ## 20   0.3413043   0.96883117    0.04422051
    ## 21   0.3444062   0.96623377    0.04459897
    ## 22   0.3473132   0.95844156    0.04479243
    ## 23   0.3503210   0.95324675    0.04510262
    ## 24   0.3531653   0.95324675    0.04574349
    ## 25   0.3560502   0.94805195    0.04614412
    ## 26   0.3590254   0.94285714    0.04653250
    ## 27   0.3618120   0.93766234    0.04677984
    ## 28   0.3643558   0.92727273    0.04698605
    ## 29   0.3670726   0.92207792    0.04739020
    ## 30   0.3696702   0.92207792    0.04797946
    ## 31   0.3721460   0.91688312    0.04847569
    ## 32   0.3746742   0.90909091    0.04871938
    ## 33   0.3771956   0.90649351    0.04929379
    ## 34   0.3800182   0.89870130    0.04958441
    ## 35   0.3826979   0.89350649    0.05005821
    ## 36   0.3851243   0.89350649    0.05083493
    ## 37   0.3876299   0.88831169    0.05143631
    ## 38   0.3901372   0.88311688    0.05192425
    ## 39   0.3927645   0.88311688    0.05275407
    ## 40   0.3952597   0.88051948    0.05342790
    ## 41   0.3976646   0.87272727    0.05389798
    ## 42   0.4001633   0.86233766    0.05401009
    ## 43   0.4025462   0.85714286    0.05479907
    ## 44   0.4050392   0.84935065    0.05521783
    ## 45   0.4076355   0.84415584    0.05580357
    ## 46   0.4100340   0.84155844    0.05668300
    ## 47   0.4123564   0.83376623    0.05716830
    ## 48   0.4148584   0.82597403    0.05759826
    ## 49   0.4174068   0.82077922    0.05842115
    ## 50   0.4199156   0.81558442    0.05936850
    ## 51   0.4228501   0.80519481    0.05967276
    ## 52   0.4258743   0.79740260    0.06039740
    ## 53   0.4284769   0.79220779    0.06109776
    ## 54   0.4310948   0.77922078    0.06137480
    ## 55   0.4337121   0.77922078    0.06285355
    ## 56   0.4362855   0.76883117    0.06349206
    ## 57   0.4392608   0.76623377    0.06472137
    ## 58   0.4421917   0.75324675    0.06509540
    ## 59   0.4449742   0.73506494    0.06507243
    ## 60   0.4477646   0.72207792    0.06553512
    ## 61   0.4505119   0.71688312    0.06655414
    ## 62   0.4535250   0.69610390    0.06645177
    ## 63   0.4566427   0.68311688    0.06690410
    ## 64   0.4598380   0.67532468    0.06799163
    ## 65   0.4629630   0.66493506    0.06887275
    ## 66   0.4660356   0.65454545    0.06961326
    ## 67   0.4695938   0.64675325    0.07098062
    ## 68   0.4731986   0.63376623    0.07170144
    ## 69   0.4763520   0.61298701    0.07145020
    ## 70   0.4795352   0.60519481    0.07285804
    ## 71   0.4832118   0.59480519    0.07418205
    ## 72   0.4869878   0.58701299    0.07543391
    ## 73   0.4909045   0.57662338    0.07689643
    ## 74   0.4949484   0.55584416    0.07695074
    ## 75   0.4989569   0.54545455    0.07835821
    ## 76   0.5032689   0.52987013    0.07962529
    ## 77   0.5076259   0.51168831    0.07982172
    ## 78   0.5120162   0.48831169    0.07966102
    ## 79   0.5164175   0.46233766    0.07900577
    ## 80   0.5205898   0.45194805    0.08111888
    ## 81   0.5252360   0.43896104    0.08276200
    ## 82   0.5309132   0.41818182    0.08298969
    ## 83   0.5371632   0.39480519    0.08278867
    ## 84   0.5437860   0.36103896    0.08048639
    ## 85   0.5503575   0.35064935    0.08266993
    ## 86   0.5568848   0.33506494    0.08492429
    ## 87   0.5638206   0.31688312    0.08670931
    ## 88   0.5713212   0.29610390    0.08715596
    ## 89   0.5801028   0.28311688    0.09090909
    ## 90   0.5895948   0.26753247    0.09423605
    ## 91   0.5988623   0.25194805    0.09709710
    ## 92   0.6105364   0.22337662    0.09641256
    ## 93   0.6234504   0.20259740    0.09936306
    ## 94   0.6370855   0.18181818    0.10233918
    ## 95   0.6548311   0.15584416    0.10695187
    ## 96   0.6772332   0.14025974    0.11250000
    ## 97   0.7047340   0.10129870    0.10655738
    ## 98   0.7424310   0.06753247    0.10196078
    ## 99   0.7991466   0.03896104    0.09868421
    ## 100  0.9152426   0.01038961    0.12121212
    ## 101        Inf   0.00000000            NA

## More Model Fitting and Performance Evaluation: naive Bayes model, filtered

We use the filtered dataset we got previously.

``` r
library(e1071)

# oversample the training set
xyzdata_normalized_drop_user_id_train_top10_oversampled <- ROSE(adopter ~ lovedTracks + delta_songsListened + delta_lovedTracks + subscriber_friend_cnt + songsListened + friend_cnt + friend_country_cnt + delta_friend_cnt + delta_subscriber_friend_cnt + delta_avg_friend_male, data = xyzdata_normalized_drop_user_id_train_top10, seed = 123)$data

# train the model
NB_model_oversample <- naiveBayes(adopter ~ lovedTracks + delta_songsListened + delta_lovedTracks + subscriber_friend_cnt + songsListened + friend_cnt + friend_country_cnt + delta_friend_cnt + delta_subscriber_friend_cnt + delta_avg_friend_male, data = xyzdata_normalized_drop_user_id_train_top10_oversampled)

# predict
preb_prob_nb_oversample <- predict(NB_model_oversample, xyzdata_normalized_drop_user_id_test_top10, type = "raw") 
# class probability predictions by setting type = "raw"
  
# get auc
library(pROC)

xyzdata_normalized_test_roc_curve_NB <- xyzdata_normalized_drop_user_id_test_top10 %>% mutate(prob = preb_prob_nb_oversample[, "1"]) %>% arrange(desc(prob)) %>% mutate(yes_1 = ifelse(adopter == "1", 1, 0)) 

roc_nb <- roc.curve(xyzdata_normalized_test_roc_curve_NB$yes_1, xyzdata_normalized_test_roc_curve_NB$prob)
```

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-24-1.png)<!-- -->

``` r
auc_nb <- roc_nb$auc
auc_nb
```

    ## [1] 0.7215105

Generate a dataframe of cutoff and corresponding recall_p and precision.

``` r
# initialize the dataframe 
dashboard_filter <- data.frame()

# initialize vectors
cutoff_filter <- c()
precision_filter <- c()
recall_p_filter <- c()

# for loop to get corresponding recall_p and precision for each cutoff value
threshold <- roc_nb$thresholds
for (i in 1:(length(threshold))){
  cutoff_filter <- c(cutoff_filter, threshold[i])
  binary_predictions <- ifelse(xyzdata_normalized_test_roc_curve_NB$prob >= threshold[i], 1, 0)
  confusion_matrix <- confusionMatrix(data = factor(binary_predictions), reference = xyzdata_normalized_drop_user_id_test_top10$adopter, mode = "prec_recall", positive = "1")
  recall_p_filter <- c(recall_p_filter, roc_nb$true.positive.rate[i])
  precision_filter <- c(precision_filter, confusion_matrix$byClass[["Precision"]]) 
}
```

    ## Warning in confusionMatrix.default(data = factor(binary_predictions), reference
    ## = xyzdata_normalized_drop_user_id_test_top10$adopter, : Levels are not in the
    ## same order for reference and data. Refactoring data to match.

    ## Warning in confusionMatrix.default(data = factor(binary_predictions), reference
    ## = xyzdata_normalized_drop_user_id_test_top10$adopter, : Levels are not in the
    ## same order for reference and data. Refactoring data to match.

``` r
dashboard_filter <- data.frame(cutoff_filter, recall_p_filter, precision_filter)
dashboard_filter
```

    ##     cutoff_filter recall_p_filter precision_filter
    ## 1            -Inf      1.00000000       0.03707270
    ## 2     0.004388274      1.00000000       0.03698353
    ## 3     0.008783567      1.00000000       0.03690685
    ## 4     0.008797732      0.99480519       0.03670449
    ## 5     0.008809540      0.98961039       0.03713316
    ## 6     0.008817218      0.97922078       0.03703330
    ## 7     0.008823178      0.97662338       0.03671596
    ## 8     0.008828817      0.97402597       0.03672462
    ## 9     0.008832889      0.97402597       0.03668295
    ## 10    0.008835353      0.97402597       0.03672524
    ## 11    0.008837797      0.96883117       0.03659185
    ## 12    0.008840473      0.96623377       0.03676233
    ## 13    0.008843052      0.96363636       0.03706119
    ## 14    0.008845307      0.96103896       0.03706148
    ## 15    0.008847068      0.96103896       0.03703704
    ## 16    0.008848669      0.96103896       0.03648893
    ## 17    0.008851079      0.96103896       0.03624883
    ## 18    0.008854578      0.95844156       0.03569318
    ## 19    0.008858289      0.95844156       0.03552632
    ## 20    0.008861319      0.95844156       0.03534681
    ## 21    0.008865138      0.95584416       0.03571429
    ## 22    0.008870750      0.95584416       0.03613855
    ## 23    0.008876492      0.95584416       0.03548715
    ## 24    0.008882229      0.95324675       0.03536977
    ## 25    0.008888796      0.95064935       0.03529105
    ## 26    0.008897615      0.94805195       0.03519915
    ## 27    0.008908319      0.94545455       0.03547998
    ## 28    0.008919181      0.94545455       0.03499254
    ## 29    0.008932539      0.94025974       0.03492849
    ## 30    0.008948730      0.93506494       0.03490645
    ## 31    0.008968197      0.92987013       0.03499575
    ## 32    0.008991776      0.92987013       0.03515065
    ## 33    0.009018071      0.92207792       0.03513120
    ## 34    0.009045975      0.91688312       0.03511878
    ## 35    0.009074932      0.90909091       0.03567128
    ## 36    0.009111347      0.90389610       0.03585050
    ## 37    0.009156121      0.89870130       0.03583062
    ## 38    0.009202195      0.89090909       0.03621477
    ## 39    0.009252973      0.88571429       0.03550296
    ## 40    0.009312711      0.87532468       0.03531901
    ## 41    0.009383507      0.87012987       0.03536606
    ## 42    0.009454303      0.86493506       0.03543241
    ## 43    0.009529206      0.85714286       0.03609923
    ## 44    0.009616163      0.84675325       0.03635415
    ## 45    0.009694769      0.84155844       0.03666962
    ## 46    0.009769615      0.84155844       0.03639640
    ## 47    0.009856876      0.83116883       0.03663476
    ## 48    0.009956358      0.82337662       0.03703011
    ## 49    0.010062875      0.81818182       0.03758108
    ## 50    0.010204187      0.80779221       0.03759252
    ## 51    0.010380941      0.80259740       0.03730159
    ## 52    0.010546535      0.78961039       0.03741909
    ## 53    0.010718492      0.78701299       0.03801653
    ## 54    0.010911655      0.78181818       0.03877766
    ## 55    0.011110145      0.76883117       0.03882657
    ## 56    0.011348512      0.76103896       0.03963885
    ## 57    0.011651430      0.75584416       0.04032440
    ## 58    0.011979965      0.74805195       0.04038772
    ## 59    0.012330211      0.72467532       0.04017013
    ## 60    0.012695122      0.71688312       0.04015481
    ## 61    0.013078206      0.71688312       0.04043662
    ## 62    0.013512044      0.69870130       0.03994911
    ## 63    0.013978128      0.69090909       0.03983338
    ## 64    0.014525377      0.68311688       0.03968892
    ## 65    0.015155688      0.66753247       0.03975704
    ## 66    0.015957362      0.65974026       0.03884321
    ## 67    0.016909663      0.64675325       0.03899884
    ## 68    0.017847238      0.63116883       0.03932753
    ## 69    0.018907631      0.62337662       0.03928240
    ## 70    0.020183324      0.60259740       0.03998720
    ## 71    0.021666186      0.59220779       0.03990765
    ## 72    0.023066901      0.57662338       0.03995936
    ## 73    0.024491672      0.56883117       0.04188853
    ## 74    0.026162655      0.55064935       0.04277286
    ## 75    0.027460678      0.54285714       0.04235025
    ## 76    0.029000681      0.53246753       0.04105221
    ## 77    0.031208188      0.51428571       0.04180464
    ## 78    0.033594740      0.50129870       0.04139715
    ## 79    0.036304538      0.49090909       0.04316547
    ## 80    0.039354195      0.47272727       0.04466385
    ## 81    0.042864029      0.45454545       0.04561230
    ## 82    0.047388086      0.43896104       0.04712042
    ## 83    0.053098552      0.41818182       0.04685777
    ## 84    0.060258286      0.40259740       0.04373178
    ## 85    0.069878150      0.38961039       0.04452690
    ## 86    0.083057863      0.36883117       0.04367968
    ## 87    0.101978285      0.35324675       0.04169611
    ## 88    0.127873443      0.32727273       0.04198473
    ## 89    0.165411915      0.31428571       0.03960396
    ## 90    0.226246497      0.28831169       0.04007124
    ## 91    0.314262575      0.26233766       0.04352127
    ## 92    0.440343870      0.22857143       0.04400440
    ## 93    0.601756790      0.19740260       0.04472050
    ## 94    0.759160260      0.16363636       0.04775281
    ## 95    0.879500919      0.13766234       0.04901961
    ## 96    0.956219579      0.11948052       0.05263158
    ## 97    0.991492379      0.10129870       0.05361305
    ## 98    0.999166663      0.07272727       0.05044510
    ## 99    0.999950678      0.04675325       0.05829596
    ## 100   0.999999610      0.03636364       0.05333333
    ## 101           Inf      0.00000000               NA

Precision and recall_p is not performed good in naive Bayes model.

## Summary and Suggestions for Business solutions outline

From the analysis above, what we will suggest for business strategy is
that, pick the model that is more explainable and feasible in business
scope rather than only focusing in numerical value of performance.

## Appendix: more model tuning and selection

### Decision tree, filtered on oversampled training set

``` r
# fit the model
# split = "information" means we want to determine splits based on information gain
library(rpart)
tree <- rpart(adopter ~ lovedTracks + delta_songsListened + delta_lovedTracks + subscriber_friend_cnt + songsListened + friend_cnt + friend_country_cnt + delta_friend_cnt + delta_subscriber_friend_cnt + delta_avg_friend_male, data = xyzdata_normalized_drop_user_id_train_top10_oversampled, method = "class", parms = list(split = "information"), control = list(minsplit = 2, maxdepth = 500, cp = 0.0005))

library(rpart.plot)
prp(tree, varlen = 0)
```

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-26-1.png)<!-- -->

``` r
pred <- predict(tree, xyzdata_normalized_drop_user_id_test_top10, type = "class")
confusion_matrix_tree <- confusionMatrix(data = factor(pred), reference = factor(xyzdata_normalized_drop_user_id_test_top10$adopter), mode = "prec_recall", positive = "1")

confusion_matrix_tree
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 9117  279
    ##          1  883  106
    ##                                           
    ##                Accuracy : 0.8881          
    ##                  95% CI : (0.8819, 0.8941)
    ##     No Information Rate : 0.9629          
    ##     P-Value [Acc > NIR] : 1               
    ##                                           
    ##                   Kappa : 0.1066          
    ##                                           
    ##  Mcnemar's Test P-Value : <2e-16          
    ##                                           
    ##               Precision : 0.10718         
    ##                  Recall : 0.27532         
    ##                      F1 : 0.15429         
    ##              Prevalence : 0.03707         
    ##          Detection Rate : 0.01021         
    ##    Detection Prevalence : 0.09523         
    ##       Balanced Accuracy : 0.59351         
    ##                                           
    ##        'Positive' Class : 1               
    ## 

Not really good performance, Precision and recall for tree model are too
low. Precision : 0.12093  
Recall : 0.29870

``` r
# auc
xyzdata_normalized_roc_curve_tree <- roc.curve(xyzdata_normalized_drop_user_id_test_top10$adopter, pred)
```

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-27-1.png)<!-- -->

``` r
# plot(xyzdata_normalized_roc_curve_knn)
xyzdata_normalized_roc_curve_tree$auc
```

    ## [1] 0.5935123

### Knn, filtered on oversampled training set

``` r
# fit the model
library(kknn)
```

    ## 
    ## Attaching package: 'kknn'

    ## The following object is masked from 'package:caret':
    ## 
    ##     contr.dummy

``` r
model_knn <- kknn(adopter ~ lovedTracks + delta_songsListened + delta_lovedTracks + subscriber_friend_cnt + songsListened + friend_cnt + friend_country_cnt + delta_friend_cnt + delta_subscriber_friend_cnt + delta_avg_friend_male, train = xyzdata_normalized_drop_user_id_train_top10_oversampled, test = xyzdata_normalized_drop_user_id_test_top10, k = 500, distance = 2, kernel = "rectangular")
pred_prob_knn <- model_knn$prob

# auc
xyzdata_normalized_roc_curve_knn <- roc.curve(ifelse(xyzdata_normalized_drop_user_id_test_top10$adopter == '1', 1, 0), pred_prob_knn[, "1"])
```

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-28-1.png)<!-- -->

``` r
# plot(xyzdata_normalized_roc_curve_knn)
xyzdata_normalized_roc_curve_knn$auc
```

    ## [1] 0.7408464

Not bad AUC. But if checking recall_p and precision, …

``` r
confusion_matrix_knn <- confusionMatrix(data = factor(ifelse(pred_prob_knn[, "1"] > 0.2, 1, 0)), reference = xyzdata_normalized_drop_user_id_test_top10$adopter, mode = "prec_recall", positive = "1")

confusion_matrix_knn
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 8231  208
    ##          1 1769  177
    ##                                           
    ##                Accuracy : 0.8096          
    ##                  95% CI : (0.8019, 0.8171)
    ##     No Information Rate : 0.9629          
    ##     P-Value [Acc > NIR] : 1               
    ##                                           
    ##                   Kappa : 0.0959          
    ##                                           
    ##  Mcnemar's Test P-Value : <2e-16          
    ##                                           
    ##               Precision : 0.09096         
    ##                  Recall : 0.45974         
    ##                      F1 : 0.15187         
    ##              Prevalence : 0.03707         
    ##          Detection Rate : 0.01704         
    ##    Detection Prevalence : 0.18739         
    ##       Balanced Accuracy : 0.64142         
    ##                                           
    ##        'Positive' Class : 1               
    ## 

Precision : 0.08541  
Recall : 0.47273  
Not performing well even if we’ve tune the threshold low.

### Oversampling before Filtering

Since the data set is severely imbalanced, we oversample the whole
training dataset, hoping the filtering procedure could capture more
information.

``` r
# oversample the whole training dataset
library(ROSE)
xyzdata_rose_whole_train <- ROSE(adopter ~., data = xyzdata_normalized_drop_user_id_train, seed = 123)$data

# xyzdata_normalized_drop_user_id_test: untouched

# filtering using oversampled training set
library(FSelectorRcpp)
IG_oversample_whole_train <- information_gain(adopter ~ ., data = xyzdata_rose_whole_train)

# select top 10
top10_oversampled <- cut_attrs(IG_oversample_whole_train, k = 10)

# oversampled training set
xyzdata_normalized_top_10_oversampled_train <- xyzdata_rose_whole_train %>% select(top10_oversampled, adopter)
```

    ## Warning: Using an external vector in selections was deprecated in tidyselect 1.1.0.
    ## ℹ Please use `all_of()` or `any_of()` instead.
    ##   # Was:
    ##   data %>% select(top10_oversampled)
    ## 
    ##   # Now:
    ##   data %>% select(all_of(top10_oversampled))
    ## 
    ## See <https://tidyselect.r-lib.org/reference/faq-external-vector.html>.
    ## This warning is displayed once every 8 hours.
    ## Call `lifecycle::last_lifecycle_warnings()` to see where this warning was
    ## generated.

``` r
# untouched testing, no oversampled
xyzdata_normalized_drop_user_id_test_top_10 <- xyzdata_normalized_drop_user_id_test %>% select(top10_oversampled, adopter)

# the whole dataset, no oversampled
xyzdata_normalized_drop_user_id_top_10 <- xyzdata_normalized_drop_user_id %>% select(top10_oversampled, adopter)

colnames(xyzdata_normalized_drop_user_id_top_10) 
```

    ##  [1] "delta_shouts"                "playlists"                  
    ##  [3] "shouts"                      "delta_posts"                
    ##  [5] "delta_playlists"             "delta_subscriber_friend_cnt"
    ##  [7] "lovedTracks"                 "delta_good_country"         
    ##  [9] "posts"                       "delta_friend_cnt"           
    ## [11] "adopter"

Yet then we realized it’s not stable since each time the top 10
variables are not exactly the same. So we decided not to use it in the
end.
