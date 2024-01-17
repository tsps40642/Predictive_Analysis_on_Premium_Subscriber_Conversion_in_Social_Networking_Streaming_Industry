Predictive Analysis on Premium Subscriber Conversion in Social
Networking Streaming Industry
================

Please note that some description parts are adjusted after rendering.  
For the plotting sections, please refer to Technical_document.pdf.

# GROUP PART – technical document

Table of contents:  
1. Things to Notice before Analysis  
1. Problem defining and our overall rationale to solve it  
2. EDA before Prediction:  
(1) due to the imbalanced dataset, oversampling to train the model is
needed  
(2) not very strong positive correlations exist; we will fit the model
later to check whether there are severe multicollinearity using Variance
Inflation Factor (VIF)  
3. Normalization: min-max normalization  
4. Before Feature Selection: with / without oversampling + no feature
selection  
5. Feature Selection: Filter Approach  
6. Model Fitting and Performance Evaluation:  
(1) fit the models with the filtered data set  
(2) 10-Folds Cross-Validation with oversampled training set within each
folds  
(2) select the final model based on AUC  
(3) generate dashboard of threshold and their corresponding precision
and recall_p  
7. Summary and other analysis results  
8. Translate analyzing results into business solutions outline  
9. Appendix: more model tuning and selection

## Things to Notice before Analysis

### Don’t use Accuracy in Imbalanced dataset; Use AUC, Precision and Recall_p

From the EDA later we will know that, the proportion of the 2 levels in
response variable is severely imbalanced, indicating we shouldn’t use
accuracy as our evaluation standard since it will conclude incorrect
results.

Thus, our priority of model selection is based on AUC, then precision
and recall_p.  
However, we are going to be not so restricted: if there’s no really big
difference in those metrices, we prefer more explainable model in
business point of view.

### Understanding the scope of our analysis: prediction task, not causal analysis

Note that all the data are originally non-subscribers.  
It is essential to emphasize our goal for understanding which people
would be likely to convert from free users to premium subscribers in the
next 6 month period if they are targeted by our promotion campaign.  
We care about correlation and can’t say “they turn into premium users
DUE TO our promotion” since it’s NOT an causal problem which statistical
method we have now cannot solve.

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

``` r
# pie chart for the response variable
pie(table(xyzdata$adopter), labels = round(table(xyzdata$adopter)/41540, 2), main = "Adopter Proportion Pie Chart", col = rainbow(2))
legend("topright", c("0: no","1: yes"), cex = 0.8, fill = rainbow(2))
```

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

Due to the severe imbalanced data, oversampling to train the model is
needed.

Check the correlation among variables to get a conceptual understanding
of the dataset.

``` r
library(corrplot)
```

    ## corrplot 0.92 loaded

``` r
corrplot(cor(xyzdata[, 2:26]), type = 'upper', addCoef.col = 'brown', tl.cex = 0.5, number.cex = 0.3)
```

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

No significant negative correlation but some positive correlations we
might want to notice later.

The followings are some reasons we consider why that positive
correlation happened:  
1. age / avg_friend_age: people generally tends to have friends with
similar age range.  
2. friend_cnt / friend_country_cnt: a person with more friends tends to
have more friends from more different countries  
3. friend_cnt / subscriber_friend_cnt: a person with more friends tends
to have more friends that are subscribers.

Check relationship between response and each predictor.

``` r
par(mfrow = c(2, 2))
plot(xyzdata$age, xyzdata$adopter, type = "p", pch = 20, cex = 0.5)
plot(xyzdata$male, xyzdata$adopter, type = "p", pch = 20, cex = 0.5)
plot(xyzdata$friend_cnt, xyzdata$adopter, type = "p", pch = 20, cex = 0.5)
plot(xyzdata$avg_friend_age, xyzdata$adopter, type = "p", pch = 20, cex = 0.5)
```

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
par(mfrow = c(2, 2))
plot(xyzdata$avg_friend_male, xyzdata$adopter, type = "p", pch = 20, cex = 0.5)
plot(xyzdata$friend_country_cnt, xyzdata$adopter, type = "p", pch = 20, cex = 0.5)
plot(xyzdata$subscriber_friend_cnt, xyzdata$adopter, type = "p", pch = 20, cex = 0.5)
plot(xyzdata$songsListened, xyzdata$adopter, type = "p", pch = 20, cex = 0.5)
```

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-5-2.png)<!-- -->

``` r
par(mfrow = c(2, 2))
plot(xyzdata$lovedTracks, xyzdata$adopter, type = "p", pch = 20, cex = 0.5)
plot(xyzdata$posts, xyzdata$adopter, type = "p", pch = 20, cex = 0.5)
plot(xyzdata$playlists, xyzdata$adopter, type = "p", pch = 20, cex = 0.5)
plot(xyzdata$shouts, xyzdata$adopter, type = "p", pch = 20, cex = 0.5)
```

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-5-3.png)<!-- -->

``` r
par(mfrow = c(2, 2))
plot(xyzdata$delta_friend_cnt, xyzdata$adopter, type = "p", pch = 20, cex = 0.5)
plot(xyzdata$delta_avg_friend_age, xyzdata$adopter, type = "p", pch = 20, cex = 0.5)
plot(xyzdata$delta_avg_friend_male, xyzdata$adopter, type = "p", pch = 20, cex = 0.5)
plot(xyzdata$delta_friend_country_cnt, xyzdata$adopter, type = "p", pch = 20, cex = 0.5)
```

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-5-4.png)<!-- -->

``` r
par(mfrow = c(2, 2))
plot(xyzdata$delta_subscriber_friend_cnt, xyzdata$adopter, type = "p", pch = 20, cex = 0.5)
plot(xyzdata$delta_songsListened, xyzdata$adopter, type = "p", pch = 20, cex = 0.5)
plot(xyzdata$delta_lovedTracks, xyzdata$adopter, type = "p", pch = 20, cex = 0.5)
plot(xyzdata$delta_posts, xyzdata$adopter, type = "p", pch = 20, cex = 0.5)
```

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-5-5.png)<!-- -->

``` r
par(mfrow = c(2, 2))
plot(xyzdata$delta_playlists, xyzdata$adopter, type = "p", pch = 20, cex = 0.5)
plot(xyzdata$delta_shouts, xyzdata$adopter, type = "p", pch = 20, cex = 0.5)
plot(xyzdata$tenure, xyzdata$adopter, type = "p", pch = 20, cex = 0.5)
plot(xyzdata$good_country, xyzdata$adopter, type = "p", pch = 20, cex = 0.5)
```

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-5-6.png)<!-- -->

``` r
par(mfrow = c(1, 2))
pie(table(xyzdata$male), labels = round(table(xyzdata$male)/41540, 2), main = "Gender Proportion Pie Chart", col = rainbow(2))
legend("topleft", c("0: female","1: male"), cex = 0.8, fill = rainbow(2))

pie(table(xyzdata$good_country), labels = round(table(xyzdata$good_country)/41540, 2), main = "Good Country Proportion Pie Chart", col = rainbow(2))
legend("topright", c("0: more limited","1: less limited"), cex = 0.8, fill = rainbow(2))
```

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-5-7.png)<!-- -->

``` r
pie(table(xyzdata$delta_good_country), labels = round(table(xyzdata$delta_good_country)/41540, 2), main = "Delta Good Country Proportion Pie Chart", col = rainbow(3))
legend("topright", c("-1: become more limited", "0: unchanged", "1: become less limited"), cex = 0.8, fill = rainbow(3))
```

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-5-8.png)<!-- -->

Variables seems to have pattern, and please note that this is a little
bit subjective judgement:  
(\>\>\>\>\> means we consider the variable acts(or ranges) more
differently in adopter_1 and adopter_0 compared to others, i.e. more
possible patterns)  
delta_shouts \>\>\>\>\>  
delta_playlist  
delta_posts \>\>\>  
delta_lovedTracks \>\>\>\>\>  
delta_songListened \>\>\>\>\>  
delta_subscriber_friend_cnt  
delta_avg_friend_age \>\>\>\>\>  
delta_avg_friend_male \>\>\>  
delta_friend_cnt \>\>\>\>\>  
posts  
lovedTracks \>\>\>\>\>  
shouts \>\>\>\>\>  
playlists  
songListened \>\>\>\>\>  
subscriber_friend_cnt  
friend_country_cnt  
avg_friend_age  
friend_cnt

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
    ## (Intercept)                  -4.72544    2.00564  -2.356 0.018469 *  
    ## age                           1.52651    0.39639   3.851 0.000118 ***
    ## male                          0.44040    0.07083   6.218 5.04e-10 ***
    ## friend_cnt                  -25.11077    6.80958  -3.688 0.000226 ***
    ## avg_friend_age                1.80835    0.50451   3.584 0.000338 ***
    ## avg_friend_male               0.06513    0.10549   0.617 0.536982    
    ## friend_country_cnt            5.21343    0.83371   6.253 4.02e-10 ***
    ## subscriber_friend_cnt         9.04074    3.10480   2.912 0.003593 ** 
    ## songsListened                 4.80185    0.87456   5.491 4.01e-08 ***
    ## lovedTracks                   5.79430    0.85185   6.802 1.03e-11 ***
    ## posts                         2.84682    2.26016   1.260 0.207826    
    ## playlists                     4.21984    1.16497   3.622 0.000292 ***
    ## shouts                        2.79293    2.45256   1.139 0.254794    
    ## delta_friend_cnt             -3.96454    3.99858  -0.991 0.321447    
    ## delta_avg_friend_age         -1.58729    1.96312  -0.809 0.418772    
    ## delta_avg_friend_male        -1.83436    1.03953  -1.765 0.077630 .  
    ## delta_friend_country_cnt      6.26418    2.54908   2.457 0.013994 *  
    ## delta_subscriber_friend_cnt  -1.83183    1.52936  -1.198 0.231004    
    ## delta_songsListened           5.28537    2.85603   1.851 0.064227 .  
    ## delta_lovedTracks            -2.70988    2.13456  -1.270 0.204254    
    ## delta_posts                   0.26847    1.78051   0.151 0.880145    
    ## delta_playlists              -1.19141    1.75323  -0.680 0.496789    
    ## delta_shouts                  6.41659    4.57464   1.403 0.160724    
    ## tenure                       -0.21233    0.18698  -1.136 0.256134    
    ## good_country                 -0.51308    0.07031  -7.297 2.94e-13 ***
    ## delta_good_country           -0.94884    1.70208  -0.557 0.577211    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 9877.8  on 31154  degrees of freedom
    ## Residual deviance: 9263.2  on 31129  degrees of freedom
    ## AIC: 9315.2
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

    ## [1] 0.7068573

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

    ##          cutoff   recall_p  precision
    ## 1          -Inf 1.00000000 0.03707270
    ## 2   0.007849006 1.00000000 0.03707984
    ## 3   0.011092412 1.00000000 0.03764913
    ## 4   0.011912318 1.00000000 0.03801718
    ## 5   0.012627139 0.99740260 0.03834249
    ## 6   0.013332118 0.99480519 0.03865563
    ## 7   0.014092447 0.98961039 0.03883792
    ## 8   0.014763187 0.98961039 0.03927835
    ## 9   0.015349404 0.98441558 0.03949151
    ## 10  0.015895264 0.97922078 0.03969257
    ## 11  0.016389515 0.97922078 0.04017477
    ## 12  0.016802377 0.97662338 0.04052161
    ## 13  0.017149233 0.97402597 0.04084967
    ## 14  0.017498113 0.97402597 0.04129501
    ## 15  0.017824971 0.97402597 0.04183400
    ## 16  0.018124340 0.97142857 0.04221219
    ## 17  0.018409054 0.96883117 0.04252651
    ## 18  0.018685865 0.96883117 0.04309150
    ## 19  0.018971066 0.96883117 0.04359514
    ## 20  0.019295719 0.96883117 0.04417861
    ## 21  0.019626672 0.96623377 0.04462572
    ## 22  0.019907207 0.96623377 0.04518402
    ## 23  0.020177290 0.96623377 0.04575646
    ## 24  0.020461819 0.96103896 0.04608295
    ## 25  0.020757020 0.95844156 0.04662034
    ## 26  0.021090984 0.95324675 0.04697299
    ## 27  0.021438152 0.94805195 0.04735340
    ## 28  0.021759274 0.93766234 0.04753753
    ## 29  0.022067200 0.93506494 0.04802561
    ## 30  0.022386675 0.93246753 0.04857259
    ## 31  0.022712133 0.92467532 0.04884072
    ## 32  0.023051555 0.91688312 0.04911646
    ## 33  0.023403145 0.90909091 0.04943503
    ## 34  0.023775665 0.90389610 0.04982818
    ## 35  0.024162853 0.89610390 0.05025492
    ## 36  0.024566862 0.89350649 0.05090263
    ## 37  0.025008234 0.89090909 0.05147059
    ## 38  0.025424004 0.88311688 0.05182927
    ## 39  0.025786847 0.87012987 0.05184956
    ## 40  0.026156936 0.86493506 0.05242443
    ## 41  0.026561100 0.85714286 0.05285074
    ## 42  0.026956361 0.84935065 0.05330073
    ## 43  0.027321361 0.83636364 0.05343511
    ## 44  0.027669514 0.82337662 0.05342997
    ## 45  0.028010671 0.82077922 0.05423962
    ## 46  0.028341640 0.81558442 0.05493352
    ## 47  0.028664562 0.81038961 0.05561497
    ## 48  0.029009963 0.80519481 0.05626134
    ## 49  0.029366984 0.78961039 0.05627545
    ## 50  0.029741721 0.78441558 0.05700264
    ## 51  0.030100971 0.77922078 0.05770340
    ## 52  0.030433291 0.76883117 0.05822187
    ## 53  0.030803292 0.76103896 0.05874098
    ## 54  0.031168280 0.75584416 0.05968007
    ## 55  0.031492443 0.73246753 0.05905759
    ## 56  0.031808367 0.71688312 0.05912596
    ## 57  0.032133650 0.70129870 0.05930156
    ## 58  0.032482216 0.69350649 0.05990577
    ## 59  0.032881081 0.68571429 0.06060606
    ## 60  0.033294354 0.67792208 0.06131078
    ## 61  0.033697115 0.66493506 0.06191052
    ## 62  0.034078400 0.66233766 0.06315007
    ## 63  0.034498740 0.65194805 0.06385144
    ## 64  0.034972415 0.63636364 0.06405229
    ## 65  0.035452397 0.63116883 0.06537530
    ## 66  0.035907856 0.61818182 0.06585501
    ## 67  0.036352836 0.61558442 0.06734868
    ## 68  0.036823503 0.60259740 0.06805515
    ## 69  0.037278712 0.59740260 0.06961259
    ## 70  0.037821511 0.57662338 0.06944010
    ## 71  0.038449703 0.57142857 0.07108239
    ## 72  0.039096322 0.55844156 0.07195448
    ## 73  0.039768813 0.54545455 0.07279029
    ## 74  0.040398505 0.53506494 0.07399425
    ## 75  0.041010265 0.52987013 0.07634731
    ## 76  0.041647646 0.51168831 0.07671340
    ## 77  0.042361087 0.49870130 0.07808052
    ## 78  0.043126585 0.48311688 0.07864693
    ## 79  0.043876598 0.46753247 0.07971656
    ## 80  0.044706679 0.44675325 0.08011178
    ## 81  0.045666462 0.43116883 0.08133268
    ## 82  0.046734693 0.41818182 0.08316116
    ## 83  0.047840800 0.40000000 0.08387800
    ## 84  0.048980049 0.37922078 0.08429561
    ## 85  0.050292559 0.37402597 0.08861538
    ## 86  0.051719864 0.35584416 0.09036939
    ## 87  0.053179940 0.34285714 0.09328622
    ## 88  0.054917004 0.32207792 0.09444021
    ## 89  0.056953865 0.30389610 0.09733777
    ## 90  0.059046087 0.28831169 0.10100091
    ## 91  0.061429815 0.25194805 0.09680639
    ## 92  0.064125327 0.23376623 0.10101010
    ## 93  0.067313953 0.23116883 0.11265823
    ## 94  0.071386895 0.21298701 0.12184250
    ## 95  0.076379435 0.18701299 0.12543554
    ## 96  0.083533484 0.16363636 0.13461538
    ## 97  0.092593863 0.12727273 0.13535912
    ## 98  0.107359187 0.09610390 0.14068441
    ## 99  0.139083976 0.05974026 0.15436242
    ## 100 0.554437559 0.01038961 0.44444444
    ## 101         Inf 0.00000000         NA

### Cross validation for logistics regression, oversampling, no feature selection

We use 10 folds cross-validation.  
Note that if we want to combind cross-validation and oversampling, we
should oversample the 9 folds as training each time INSIDE the loop.

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

    ## [1] 0.7485831

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
    ##                    1.294512                    1.019985 
    ##                  friend_cnt              avg_friend_age 
    ##                    1.249496                    1.310725 
    ##             avg_friend_male          friend_country_cnt 
    ##                    1.018205                    1.307021 
    ##       subscriber_friend_cnt               songsListened 
    ##                    1.147007                    1.191234 
    ##                 lovedTracks                       posts 
    ##                    1.060157                    1.035925 
    ##                   playlists                      shouts 
    ##                    1.019912                    1.093015 
    ##            delta_friend_cnt        delta_avg_friend_age 
    ##                    1.149467                    1.027800 
    ##       delta_avg_friend_male    delta_friend_country_cnt 
    ##                    1.027094                    1.118230 
    ## delta_subscriber_friend_cnt         delta_songsListened 
    ##                    1.021842                    1.135874 
    ##           delta_lovedTracks                 delta_posts 
    ##                    1.052454                    1.017668 
    ##             delta_playlists                delta_shouts 
    ##                    1.016082                    1.023719 
    ##                      tenure                good_country 
    ##                    1.119794                    1.018049 
    ##          delta_good_country 
    ##                    1.009092

Variance Inflation Factors: VIF = 1/(1 - R_squared^2), detects
multicollinearity in regression analysis.  
Multicollinearity happens when independent variables in a regression
model are highly correlated to each other, making it hard to interpret
the model and also causes problems in performance.

Reading VIF:  
VIF of 1.9 indicates the variance of a particular coefficient is 90%
higher than what we would expect if there was no multicollinearity,
i.e. the variance of a particular coefficient is 90% higher than being
orthogonal.

Usually VIF \< 2 is not going to cause problems, which is the case
here.  
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

So far, we’ve seen some important points:  
1. oversampling is needed.  
2. no severe multicollinearity problems needed to concern, but we will
still try PCA later.  
3. filtered as top 10 variables act in relatively clear pattern in EDA
previously.

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
    ## Standard deviation     0.1265 0.08123 0.07113 0.04712 0.04283 0.02155 0.01736
    ## Proportion of Variance 0.4801 0.19810 0.15189 0.06666 0.05508 0.01394 0.00905
    ## Cumulative Proportion  0.4801 0.67821 0.83011 0.89677 0.95185 0.96579 0.97484
    ##                            PC8     PC9    PC10    PC11    PC12    PC13
    ## Standard deviation     0.01398 0.01359 0.01242 0.01115 0.01010 0.00879
    ## Proportion of Variance 0.00586 0.00555 0.00463 0.00373 0.00306 0.00232
    ## Cumulative Proportion  0.98071 0.98626 0.99088 0.99462 0.99768 1.00000

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

    ##                                    PC1         PC2           PC3           PC4
    ## age                       0.8644251738  0.01804478  0.5022742172 -0.0089189856
    ## friend_cnt               -0.0052796202  0.09277921  0.0006693621 -0.0046591953
    ## avg_friend_age            0.5022139905 -0.01235489 -0.8642157361 -0.0137566668
    ## friend_country_cnt       -0.0123312478  0.92654021 -0.0186937723 -0.3234988739
    ## subscriber_friend_cnt     0.0088406160  0.08724685 -0.0034804933 -0.0007075816
    ## songsListened            -0.0009740718  0.23504909  0.0057548839  0.4271669825
    ## lovedTracks               0.0137911507  0.23189904 -0.0194129313  0.8396642933
    ## shouts                   -0.0045629050  0.07619790  0.0013447346  0.0084289417
    ## delta_friend_cnt         -0.0051882807  0.05896407  0.0040317511 -0.0003394865
    ## delta_friend_country_cnt -0.0030730121  0.05265971  0.0056215311 -0.0054028002
    ## delta_songsListened      -0.0044715513  0.03880641  0.0034652975  0.0568712084
    ## delta_lovedTracks        -0.0044142113  0.04076177  0.0033833074  0.0648159949
    ## delta_shouts             -0.0020807548  0.02400760  0.0025792777 -0.0037308453
    ##                                   PC5           PC6          PC7          PC8
    ## age                       0.002556123  0.0003793766 -0.001321302 -0.002348650
    ## friend_cnt               -0.003706936 -0.1022944952 -0.173627166  0.277883356
    ## avg_friend_age           -0.017263183 -0.0120347095 -0.001539484 -0.000421568
    ## friend_country_cnt        0.099899514  0.1135970895  0.076459560 -0.061339758
    ## subscriber_friend_cnt     0.002466304 -0.1659380030 -0.152589974  0.614947701
    ## songsListened            -0.863719579  0.0213912739  0.037735234 -0.037645884
    ## lovedTracks               0.481407072  0.0530483607  0.013844822 -0.056944060
    ## shouts                   -0.021758838 -0.1897062201 -0.908707351 -0.260557808
    ## delta_friend_cnt          0.018626413 -0.3944480869 -0.000563434  0.145841509
    ## delta_friend_country_cnt  0.021356489 -0.8071110020  0.298940692 -0.359997934
    ## delta_songsListened      -0.095010524 -0.1273065005  0.076828382  0.217838193
    ## delta_lovedTracks         0.039479000 -0.2172738724  0.006856110  0.519587806
    ## delta_shouts              0.008327733 -0.1981379063 -0.134311538  0.002910619
    ##                                   PC9          PC10         PC11         PC12
    ## age                      -0.006305844 -0.0004655493 -0.004599983 -0.003067950
    ## friend_cnt                0.218807846 -0.0517062215 -0.436605681 -0.731964363
    ## avg_friend_age           -0.009038423 -0.0010084141 -0.005862662 -0.002633249
    ## friend_country_cnt       -0.047873540 -0.0058565893  0.022221298  0.031865494
    ## subscriber_friend_cnt     0.577710476  0.1654095890  0.397666328  0.197763861
    ## songsListened             0.047260401 -0.0976810532 -0.007153581  0.032855443
    ## lovedTracks               0.038658927  0.0319276534 -0.008690769  0.004745990
    ## shouts                   -0.137037046  0.0224054305  0.140393833 -0.038506161
    ## delta_friend_cnt          0.028881570 -0.0151819013 -0.691231937  0.338259341
    ## delta_friend_country_cnt  0.117554462 -0.0325926195  0.269343747 -0.191606865
    ## delta_songsListened      -0.466357550  0.8228235651  0.032493243 -0.142078597
    ## delta_lovedTracks        -0.598320364 -0.5248409106  0.202494483 -0.026188819
    ## delta_shouts             -0.057648687  0.0712619466 -0.196068997  0.499496133
    ##                                   PC13
    ## age                       0.0002738927
    ## friend_cnt                0.3107278131
    ## avg_friend_age            0.0009826868
    ## friend_country_cnt       -0.0064797479
    ## subscriber_friend_cnt    -0.0703744131
    ## songsListened             0.0054939663
    ## lovedTracks               0.0025512321
    ## shouts                   -0.1533772448
    ## delta_friend_cnt         -0.4753819969
    ## delta_friend_country_cnt  0.0465599164
    ## delta_songsListened      -0.0294546991
    ## delta_lovedTracks         0.0136492878
    ## delta_shouts              0.8035431647

(will be replaced with the labeled picture PDF file after rendering.)

``` r
xyzdata_rose_pca_high_cor_score <- xyzdata_rose_pca_high_cor$x

# plot the relationship 
par(mfrow = c(1, 2))
plot(xyzdata_rose_pca_high_cor_score[, "PC1"], high_cor$age, pch = 20, cex = 0.8)
plot(xyzdata_rose_pca_high_cor_score[, "PC1"], high_cor$avg_friend_age, pch = 20, cex = 0.8)
```

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

``` r
par(mfrow = c(1, 1))
plot(xyzdata_rose_pca_high_cor_score[, "PC2"], high_cor$friend_country_cnt, pch = 20, cex = 0.8)
```

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-19-2.png)<!-- -->

``` r
par(mfrow = c(1, 2))
plot(xyzdata_rose_pca_high_cor_score[, "PC3"], high_cor$age, pch = 20, cex = 0.8)
plot(xyzdata_rose_pca_high_cor_score[, "PC3"], high_cor$avg_friend_age, pch = 20, cex = 0.8)
```

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-19-3.png)<!-- -->

``` r
par(mfrow = c(1, 2))
plot(xyzdata_rose_pca_high_cor_score[, "PC4"], high_cor$songsListened, pch = 20, cex = 0.8)
plot(xyzdata_rose_pca_high_cor_score[, "PC4"], high_cor$lovedTracks, pch = 20, cex = 0.8)
```

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-19-4.png)<!-- -->

``` r
par(mfrow = c(1, 2))
plot(xyzdata_rose_pca_high_cor_score[, "PC5"], high_cor$songsListened, pch = 20, cex = 0.8)
plot(xyzdata_rose_pca_high_cor_score[, "PC5"], high_cor$lovedTracks, pch = 20, cex = 0.8)
```

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-19-5.png)<!-- -->

``` r
par(mfrow = c(2, 2))
plot(xyzdata_rose_pca_high_cor_score[, "PC6"], high_cor$delta_friend_cnt, pch = 20, cex = 0.8)
plot(xyzdata_rose_pca_high_cor_score[, "PC6"], high_cor$delta_friend_country_cnt, pch = 20, cex = 0.8)
plot(xyzdata_rose_pca_high_cor_score[, "PC6"], high_cor$delta_shouts, pch = 20, cex = 0.8)

# identify the PCs, give up PC3
age_related <- xyzdata_rose_pca_high_cor_score[, "PC1"]
friend_diversity <- -1*xyzdata_rose_pca_high_cor_score[, "PC2"]
tend_to_exploration <- xyzdata_rose_pca_high_cor_score[, "PC4"]
tend_to_concentration <- xyzdata_rose_pca_high_cor_score[, "PC5"]
information_received <- xyzdata_rose_pca_high_cor_score[, "PC6"]

# fit the logistic regression
PC1 <- scale(age_related)
PC2 <- scale(friend_diversity)
PC4 <- scale(tend_to_exploration)
PC5 <- scale(tend_to_concentration)
PC6 <- scale(information_received)
```

![](Predictive_Analysis_on_Premium_Subscriber_Conversion_in_Social_Networking_Streaming_Industry_files/figure-gfm/unnamed-chunk-19-6.png)<!-- -->

Also consider the plots above:  
Until now, we decide to extract the first 6 PCs except for the 3rd since
the variable it measures is already in PC1 and its magnitude is too
small and important information may already be included in PC1.  
PC1: capture age_related information, maybe usage and age, including
age, avg_friend_age.  
PC2: capture friend_diversity information, including
friend_country_cnt.  
PC4: capture tend_to_exploration information, meaning that users might
have many already loved content and also tend to try new things,
including songsListened, lovedTracks  
PC5: capture tend_to_concentration information, meaning that users might
have many already loved content but not tend to try new things,
including songsListened, lovedTracks  
PC6: capture information_received ability since delta_shots is the
changed number of wall posts received, and increase in number of friends
and friend’s countries will also have effects on information_received
ability, including delta_shouts, delta_friend_cnt,
delta_friend_country_cnt.

Thus we identify the 5 PCs:  
PC1(age_related), PC2(friend_diversity), PC4(tend_to_exploration),
PC5(tend_to_concentration), PC6(information_received).

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
    ##                    1.294512                    1.019985 
    ##                  friend_cnt              avg_friend_age 
    ##                    1.249496                    1.310725 
    ##             avg_friend_male          friend_country_cnt 
    ##                    1.018205                    1.307021 
    ##       subscriber_friend_cnt               songsListened 
    ##                    1.147007                    1.191234 
    ##                 lovedTracks                       posts 
    ##                    1.060157                    1.035925 
    ##                   playlists                      shouts 
    ##                    1.019912                    1.093015 
    ##            delta_friend_cnt        delta_avg_friend_age 
    ##                    1.149467                    1.027800 
    ##       delta_avg_friend_male    delta_friend_country_cnt 
    ##                    1.027094                    1.118230 
    ## delta_subscriber_friend_cnt         delta_songsListened 
    ##                    1.021842                    1.135874 
    ##           delta_lovedTracks                 delta_posts 
    ##                    1.052454                    1.017668 
    ##             delta_playlists                delta_shouts 
    ##                    1.016082                    1.023719 
    ##                      tenure                good_country 
    ##                    1.119794                    1.018049 
    ##          delta_good_country 
    ##                    1.009092

``` r
vif(fit_PCA)
```

    ##                         PC1                         PC2 
    ##                    1.069424                    1.100098 
    ##                         PC4                         PC5 
    ##                    1.058891                    1.046309 
    ##                         PC6                        male 
    ##                    1.071743                    1.012247 
    ##             avg_friend_male                       posts 
    ##                    1.006418                    1.041889 
    ##                   playlists        delta_avg_friend_age 
    ##                    1.015311                    1.017152 
    ##       delta_avg_friend_male delta_subscriber_friend_cnt 
    ##                    1.017277                    1.019478 
    ##                 delta_posts             delta_playlists 
    ##                    1.033655                    1.012793 
    ##                      tenure                good_country 
    ##                    1.097496                    1.016477 
    ##          delta_good_country 
    ##                    1.008354

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
    ## (Intercept)                  -4.72544    2.00564  -2.356 0.018469 *  
    ## age                           1.52651    0.39639   3.851 0.000118 ***
    ## male                          0.44040    0.07083   6.218 5.04e-10 ***
    ## friend_cnt                  -25.11077    6.80958  -3.688 0.000226 ***
    ## avg_friend_age                1.80835    0.50451   3.584 0.000338 ***
    ## avg_friend_male               0.06513    0.10549   0.617 0.536982    
    ## friend_country_cnt            5.21343    0.83371   6.253 4.02e-10 ***
    ## subscriber_friend_cnt         9.04074    3.10480   2.912 0.003593 ** 
    ## songsListened                 4.80185    0.87456   5.491 4.01e-08 ***
    ## lovedTracks                   5.79430    0.85185   6.802 1.03e-11 ***
    ## posts                         2.84682    2.26016   1.260 0.207826    
    ## playlists                     4.21984    1.16497   3.622 0.000292 ***
    ## shouts                        2.79293    2.45256   1.139 0.254794    
    ## delta_friend_cnt             -3.96454    3.99858  -0.991 0.321447    
    ## delta_avg_friend_age         -1.58729    1.96312  -0.809 0.418772    
    ## delta_avg_friend_male        -1.83436    1.03953  -1.765 0.077630 .  
    ## delta_friend_country_cnt      6.26418    2.54908   2.457 0.013994 *  
    ## delta_subscriber_friend_cnt  -1.83183    1.52936  -1.198 0.231004    
    ## delta_songsListened           5.28537    2.85603   1.851 0.064227 .  
    ## delta_lovedTracks            -2.70988    2.13456  -1.270 0.204254    
    ## delta_posts                   0.26847    1.78051   0.151 0.880145    
    ## delta_playlists              -1.19141    1.75323  -0.680 0.496789    
    ## delta_shouts                  6.41659    4.57464   1.403 0.160724    
    ## tenure                       -0.21233    0.18698  -1.136 0.256134    
    ## good_country                 -0.51308    0.07031  -7.297 2.94e-13 ***
    ## delta_good_country           -0.94884    1.70208  -0.557 0.577211    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 9877.8  on 31154  degrees of freedom
    ## Residual deviance: 9263.2  on 31129  degrees of freedom
    ## AIC: 9315.2
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
    ## (Intercept)                  0.71545    0.47256   1.514 0.130028    
    ## PC1                          2.26730    0.10114  22.418  < 2e-16 ***
    ## PC2                          6.65825    0.18703  35.600  < 2e-16 ***
    ## PC4                          7.11291    0.31769  22.389  < 2e-16 ***
    ## PC5                         -1.23819    0.32277  -3.836 0.000125 ***
    ## PC6                         -9.46664    0.71532 -13.234  < 2e-16 ***
    ## male                         0.30269    0.02135  14.176  < 2e-16 ***
    ## avg_friend_male              0.06879    0.03408   2.019 0.043527 *  
    ## posts                        2.48501    1.23854   2.006 0.044814 *  
    ## playlists                    6.19449    0.63909   9.693  < 2e-16 ***
    ## delta_avg_friend_age        -1.07711    0.68255  -1.578 0.114552    
    ## delta_avg_friend_male       -0.50239    0.31960  -1.572 0.115964    
    ## delta_subscriber_friend_cnt  0.07599    0.48432   0.157 0.875317    
    ## delta_posts                  0.50396    0.92291   0.546 0.585029    
    ## delta_playlists              0.40242    0.63255   0.636 0.524654    
    ## tenure                      -0.09087    0.05551  -1.637 0.101656    
    ## good_country                -0.38172    0.02166 -17.621  < 2e-16 ***
    ## delta_good_country          -0.50812    0.38600  -1.316 0.188053    
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

    ## Area under the curve (AUC): 0.727

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
    ## 2    0.1497619   1.00000000    0.03707627
    ## 3    0.2487574   1.00000000    0.03760867
    ## 4    0.2622048   1.00000000    0.03799092
    ## 5    0.2722010   0.99740260    0.03833100
    ## 6    0.2817523   0.99740260    0.03874092
    ## 7    0.2896870   0.99740260    0.03914373
    ## 8    0.2960496   0.99740260    0.03958355
    ## 9    0.3015804   0.99480519    0.03987922
    ## 10   0.3060447   0.99220779    0.04023170
    ## 11   0.3099361   0.99220779    0.04064695
    ## 12   0.3137958   0.98961039    0.04103835
    ## 13   0.3178978   0.98701299    0.04138983
    ## 14   0.3216580   0.98441558    0.04179993
    ## 15   0.3250805   0.98441558    0.04226609
    ## 16   0.3284680   0.98441558    0.04271385
    ## 17   0.3317347   0.98441558    0.04325990
    ## 18   0.3351272   0.98441558    0.04379984
    ## 19   0.3382769   0.98181818    0.04425193
    ## 20   0.3413682   0.98181818    0.04477082
    ## 21   0.3445489   0.97662338    0.04507853
    ## 22   0.3476675   0.97662338    0.04567541
    ## 23   0.3508052   0.97662338    0.04623709
    ## 24   0.3539063   0.97402597    0.04669406
    ## 25   0.3573335   0.97142857    0.04723415
    ## 26   0.3607020   0.96363636    0.04747888
    ## 27   0.3638063   0.95844156    0.04790963
    ## 28   0.3667791   0.95584416    0.04842105
    ## 29   0.3697439   0.95324675    0.04888770
    ## 30   0.3726967   0.95324675    0.04958119
    ## 31   0.3755579   0.94805195    0.05010983
    ## 32   0.3783907   0.94545455    0.05065405
    ## 33   0.3810449   0.93766234    0.05096710
    ## 34   0.3841477   0.93246753    0.05149168
    ## 35   0.3871344   0.92207792    0.05167394
    ## 36   0.3897094   0.91428571    0.05200946
    ## 37   0.3925365   0.90909091    0.05245017
    ## 38   0.3952942   0.90129870    0.05296901
    ## 39   0.3978789   0.89090909    0.05326087
    ## 40   0.4005589   0.88311688    0.05367009
    ## 41   0.4032046   0.87532468    0.05398911
    ## 42   0.4057030   0.86233766    0.05414220
    ## 43   0.4081631   0.85714286    0.05472637
    ## 44   0.4105368   0.84935065    0.05513404
    ## 45   0.4129548   0.83896104    0.05551736
    ## 46   0.4156382   0.83636364    0.05634296
    ## 47   0.4181982   0.81818182    0.05613973
    ## 48   0.4206420   0.81038961    0.05669635
    ## 49   0.4229987   0.80779221    0.05757127
    ## 50   0.4255005   0.80779221    0.05877906
    ## 51   0.4281394   0.80259740    0.05950318
    ## 52   0.4304518   0.80000000    0.06047516
    ## 53   0.4331469   0.78961039    0.06109325
    ## 54   0.4358615   0.78441558    0.06187257
    ## 55   0.4383182   0.77922078    0.06278778
    ## 56   0.4410084   0.77142857    0.06362468
    ## 57   0.4435742   0.76623377    0.06460797
    ## 58   0.4459363   0.75324675    0.06508079
    ## 59   0.4485997   0.73766234    0.06530237
    ## 60   0.4515169   0.72987013    0.06625796
    ## 61   0.4544094   0.71688312    0.06663448
    ## 62   0.4573971   0.70389610    0.06697973
    ## 63   0.4603110   0.69870130    0.06843042
    ## 64   0.4634430   0.68051948    0.06835377
    ## 65   0.4668300   0.66493506    0.06894694
    ## 66   0.4702408   0.65454545    0.06974813
    ## 67   0.4736897   0.64155844    0.07043057
    ## 68   0.4770805   0.62597403    0.07077827
    ## 69   0.4804142   0.61038961    0.07125531
    ## 70   0.4837584   0.60259740    0.07240949
    ## 71   0.4872741   0.60000000    0.07454017
    ## 72   0.4908971   0.58961039    0.07589435
    ## 73   0.4950533   0.58441558    0.07823366
    ## 74   0.4995815   0.57142857    0.07905138
    ## 75   0.5041277   0.55064935    0.07934132
    ## 76   0.5089727   0.52987013    0.07947020
    ## 77   0.5136221   0.51428571    0.08045510
    ## 78   0.5183776   0.49090909    0.08015267
    ## 79   0.5231365   0.47532468    0.08126110
    ## 80   0.5276854   0.45974026    0.08213457
    ## 81   0.5333069   0.45454545    0.08553275
    ## 82   0.5397073   0.44155844    0.08799172
    ## 83   0.5457146   0.42077922    0.08813928
    ## 84   0.5516948   0.40259740    0.08975101
    ## 85   0.5583654   0.39220779    0.09292308
    ## 86   0.5649196   0.37142857    0.09395532
    ## 87   0.5715329   0.35584416    0.09730114
    ## 88   0.5792816   0.32727273    0.09618321
    ## 89   0.5884649   0.31428571    0.10100167
    ## 90   0.5982297   0.30129870    0.10583942
    ## 91   0.6078583   0.27532468    0.10696266
    ## 92   0.6190087   0.25454545    0.10998878
    ## 93   0.6320405   0.22597403    0.11111111
    ## 94   0.6478319   0.20259740    0.11641791
    ## 95   0.6670464   0.18181818    0.12048193
    ## 96   0.6890455   0.15844156    0.12978723
    ## 97   0.7167485   0.13506494    0.14444444
    ## 98   0.7579200   0.09870130    0.14901961
    ## 99   0.8154300   0.04415584    0.11564626
    ## 100  0.9237544   0.01558442    0.15789474
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

    ## [1] 0.7301351

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
    ## 2     0.004956769      1.00000000       0.03707627
    ## 3     0.009921977      0.99480519       0.03750733
    ## 4     0.009937935      0.99220779       0.03687999
    ## 5     0.009953046      0.98961039       0.03662309
    ## 6     0.009968507      0.98441558       0.03675360
    ## 7     0.009982404      0.98181818       0.03683354
    ## 8     0.009993685      0.98181818       0.03698757
    ## 9     0.010003335      0.98181818       0.03691797
    ## 10    0.010011472      0.97922078       0.03705259
    ## 11    0.010018487      0.97662338       0.03710772
    ## 12    0.010024425      0.97662338       0.03721978
    ## 13    0.010030088      0.97142857       0.03718963
    ## 14    0.010034956      0.96883117       0.03741086
    ## 15    0.010038158      0.96883117       0.03699178
    ## 16    0.010041153      0.96623377       0.03696629
    ## 17    0.010044098      0.96103896       0.03683911
    ## 18    0.010046964      0.95844156       0.03693901
    ## 19    0.010049770      0.95324675       0.03685131
    ## 20    0.010052133      0.95324675       0.03708077
    ## 21    0.010054289      0.95324675       0.03704146
    ## 22    0.010056821      0.95064935       0.03747977
    ## 23    0.010060365      0.95064935       0.03759115
    ## 24    0.010065016      0.95064935       0.03781837
    ## 25    0.010070475      0.94545455       0.03780472
    ## 26    0.010077942      0.94545455       0.03795292
    ## 27    0.010088692      0.94545455       0.03806971
    ## 28    0.010103056      0.94285714       0.03816898
    ## 29    0.010119050      0.94285714       0.03772026
    ## 30    0.010136373      0.93506494       0.03725408
    ## 31    0.010156046      0.93506494       0.03724157
    ## 32    0.010180171      0.92467532       0.03697310
    ## 33    0.010210173      0.91688312       0.03667588
    ## 34    0.010244175      0.91428571       0.03613744
    ## 35    0.010280401      0.91168831       0.03590745
    ## 36    0.010321669      0.90909091       0.03619426
    ## 37    0.010367658      0.90909091       0.03594670
    ## 38    0.010415442      0.90649351       0.03601195
    ## 39    0.010474823      0.89870130       0.03582853
    ## 40    0.010547292      0.89610390       0.03556927
    ## 41    0.010644125      0.88831169       0.03590338
    ## 42    0.010744222      0.88311688       0.03579832
    ## 43    0.010822529      0.87792208       0.03572039
    ## 44    0.010897316      0.87012987       0.03495652
    ## 45    0.010966116      0.86233766       0.03524619
    ## 46    0.011040841      0.84935065       0.03492349
    ## 47    0.011146931      0.84675325       0.03503944
    ## 48    0.011281902      0.83896104       0.03499251
    ## 49    0.011409702      0.83376623       0.03521797
    ## 50    0.011551853      0.82857143       0.03568658
    ## 51    0.011732756      0.82337662       0.03538948
    ## 52    0.011931015      0.82077922       0.03577203
    ## 53    0.012133586      0.81818182       0.03611971
    ## 54    0.012317493      0.80259740       0.03640572
    ## 55    0.012538397      0.79740260       0.03657487
    ## 56    0.012820613      0.78181818       0.03677604
    ## 57    0.013101843      0.76363636       0.03734533
    ## 58    0.013396816      0.74805195       0.03771849
    ## 59    0.013785863      0.72987013       0.03788235
    ## 60    0.014220870      0.72207792       0.03741250
    ## 61    0.014637902      0.71428571       0.03748459
    ## 62    0.015068890      0.70129870       0.03730018
    ## 63    0.015586886      0.68051948       0.03714286
    ## 64    0.016222670      0.67012987       0.03658211
    ## 65    0.016926166      0.64935065       0.03619413
    ## 66    0.017764790      0.64935065       0.03584533
    ## 67    0.018720357      0.63636364       0.03576621
    ## 68    0.019787808      0.62077922       0.03588517
    ## 69    0.020989642      0.61298701       0.03570329
    ## 70    0.022328943      0.60259740       0.03601020
    ## 71    0.023733775      0.59220779       0.03625577
    ## 72    0.025581712      0.58441558       0.03675970
    ## 73    0.027797522      0.57402597       0.03608970
    ## 74    0.029466375      0.56883117       0.03656307
    ## 75    0.030995557      0.56623377       0.03799392
    ## 76    0.032871490      0.55064935       0.03766183
    ## 77    0.034694031      0.54025974       0.03677686
    ## 78    0.036968595      0.52467532       0.03716361
    ## 79    0.039985255      0.51428571       0.03791258
    ## 80    0.043299868      0.49870130       0.03928906
    ## 81    0.047445280      0.48051948       0.03974485
    ## 82    0.052502280      0.45194805       0.03921569
    ## 83    0.058195091      0.42857143       0.03978202
    ## 84    0.065046304      0.40519481       0.03912543
    ## 85    0.073953812      0.38701299       0.03787416
    ## 86    0.086777446      0.37402597       0.04052288
    ## 87    0.106861378      0.36103896       0.03910615
    ## 88    0.134822654      0.34805195       0.04054054
    ## 89    0.167955211      0.32987013       0.03748981
    ## 90    0.212726843      0.31428571       0.03700441
    ## 91    0.291626280      0.29870130       0.03678606
    ## 92    0.407327433      0.27012987       0.03952991
    ## 93    0.552117060      0.24415584       0.03823178
    ## 94    0.715968384      0.20779221       0.03527815
    ## 95    0.862582356      0.18701299       0.03726708
    ## 96    0.954498526      0.15584416       0.03717472
    ## 97    0.990454060      0.14025974       0.03555556
    ## 98    0.998937880      0.10649351       0.02785515
    ## 99    0.999924822      0.08831169       0.03284672
    ## 100   0.999999841      0.06233766       0.04000000
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
    ##          0 9167  290
    ##          1  833   95
    ##                                           
    ##                Accuracy : 0.8919          
    ##                  95% CI : (0.8857, 0.8978)
    ##     No Information Rate : 0.9629          
    ##     P-Value [Acc > NIR] : 1               
    ##                                           
    ##                   Kappa : 0.0974          
    ##                                           
    ##  Mcnemar's Test P-Value : <2e-16          
    ##                                           
    ##               Precision : 0.102371        
    ##                  Recall : 0.246753        
    ##                      F1 : 0.144707        
    ##              Prevalence : 0.037073        
    ##          Detection Rate : 0.009148        
    ##    Detection Prevalence : 0.089360        
    ##       Balanced Accuracy : 0.581727        
    ##                                           
    ##        'Positive' Class : 1               
    ## 

Not really good performance, Precision and recall for tree model are too
low.  
Precision : 0.12093  
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

    ## [1] 0.5817266

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

    ## [1] 0.7239608

Not bad AUC. But if checking recall_p and precision, …

``` r
confusion_matrix_knn <- confusionMatrix(data = factor(ifelse(pred_prob_knn[, "1"] > 0.2, 1, 0)), reference = xyzdata_normalized_drop_user_id_test_top10$adopter, mode = "prec_recall", positive = "1")

confusion_matrix_knn
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 8050  204
    ##          1 1950  181
    ##                                           
    ##                Accuracy : 0.7926          
    ##                  95% CI : (0.7847, 0.8003)
    ##     No Information Rate : 0.9629          
    ##     P-Value [Acc > NIR] : 1               
    ##                                           
    ##                   Kappa : 0.0865          
    ##                                           
    ##  Mcnemar's Test P-Value : <2e-16          
    ##                                           
    ##               Precision : 0.08494         
    ##                  Recall : 0.47013         
    ##                      F1 : 0.14388         
    ##              Prevalence : 0.03707         
    ##          Detection Rate : 0.01743         
    ##    Detection Prevalence : 0.20520         
    ##       Balanced Accuracy : 0.63756         
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

    ##  [1] "delta_shouts"                "delta_good_country"         
    ##  [3] "delta_posts"                 "playlists"                  
    ##  [5] "shouts"                      "lovedTracks"                
    ##  [7] "delta_subscriber_friend_cnt" "posts"                      
    ##  [9] "subscriber_friend_cnt"       "delta_playlists"            
    ## [11] "adopter"

Yet then we realized it’s not stable since each time the top 10
variables are not exactly the same. So we decided not to use it in the
end.
