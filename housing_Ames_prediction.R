---
title: "Predicting Residential House Price in Ames,Iowa"
author: "Dan Derieg(dderi2@illinois.edu)"
date: "08/07/2020"
output:
  pdf_document: default
  html_document: 
    toc: yes
urlcolor: cyan
---


# Introduction

*The purpose of this project is to predict the residential house price in Ames, Iowa.*
To do this we create a linear model which can predict house prices in Iowa based on diﬀerent predictor variables. This approach is similar to how companies like Redﬁn and Zillow calculate their price estimate for their websites

Our final project will try to touch some of the concepts mentioned below:

- Multiple linear regression
- Dummy variables
- Interaction
- Residual diagnostics
- Outlier diagnostics
- Transformations
- Polynomial regression
- Model selection


## Dataset Source

The dataset is provided by Dean De Cock from Truman State University and initiative was inspired by Kaggle problem https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview.  

The data consists of 2930 observations and 82 variables. It includes of a mix of nominal, discrete, ordinal and continuous variables. 

The raw dataset is rather large consisting of 82 potential features. As such we make a subset of the data consisting of the 25 predictors we consider the most relevant to our study. The dataset is further refined to only include residential properties.

## Loding Data

We loaded `housing_data` into R.


Below are a few of the variables we considered to be important for our regression analysis.

- `MS Zoning`: Identifies the general zoning classification of the sale.
- `Lot Area`:  Lot size in square feet.
- `Overall Qual`: Overall material and finish quality.
- `Total Bsmt SF`: Total square feet of basement area.
- `Year Built`: Original construction date.
- `Bedroom AbvGr`: Bedrooms above grade (does NOT include basement bedrooms).
- `Bsmt Full Bath`:  Full bathrooms above grade.
- `GarageCars`: Size of garage in car capacity.
- `Street`: Type of road access to property


```{r message=FALSE, warning=FALSE}
#Install all packages
library(readr)
library(stringr)
library(ggplot2)
library(dplyr)
library(corrplot)
library(tidyverse)
library("VIM")
library(car)
library(lmtest)
library(caret)
library(rpart)
library(rpart.plot)
library(PerformanceAnalytics)
library(trafo)
#install.packages("stringr")
#install.packages("corrplot")
#install.packages("tidyverse")
#install.packages("car")
```


## Data Sanity Checking

```{r message=FALSE, warning=FALSE}
housing_data = read_csv("housing_data.csv")
names(housing_data) = str_replace_all(names(housing_data), c(" " = "." , "," = "" ))
```

```{r}
housing_data[c(1:10),c(1:5)]
```



# Methods

To begin the analysis, data cleaning techniques were applied. On the basis of this some predictors were selected and a subset created as opposed to using all of the variables. The data is also examined for variables which have too many ’NA’s, which could impact the results.

## Data Cleansing and Preprocessing

### Analyse Missing Values for predictors


```{r, eval = FALSE}
sapply(housing_data, function(x) sum(is.na(x)))
```

Below is the graphical representation of NA values.
```{r}
missing_data = housing_data[sapply(housing_data, function(x) sum(is.na(x))) > 100]
aggr(missing_data)
```

### Data Transformations 

2 columns which have a lot of `NAs` are `Alley` and `Lot Frontage`.

- 2732 out of 2930 records for `Alley` were ’NA’s, making it useless as a predictor, so it was removed. Other columns such as `Fireplace.Qu`,`Pool.QC`,`Fence`,`Misc.Feature` contained a lot of NA values and were removed also.

- Lot Frontage is an important feature and replaces the missing values with the mean of the data. SWe will do the same with `Garage.Cars` and `Total.Bsmt.SF` as well.

```{r}
NA_values=data.frame(no_of_na_values=colSums(is.na(housing_data)))
#Replace NA values  with mean
housing_data$Lot.Frontage[which(is.na(housing_data$Lot.Frontage))] = mean(housing_data$Lot.Frontage,na.rm = TRUE)
housing_data$Total.Bsmt.SF[which(is.na(housing_data$Total.Bsmt.SF))] = mean(housing_data$Total.Bsmt.SF,na.rm = TRUE)
housing_data$Garage.Area[which(is.na(housing_data$Garage.Area))] = mean(housing_data$Garage.Area,na.rm = TRUE)
```

### Data Creation with Relevant predictors
                   
```{r}
names(housing_data)[names(housing_data) == "Year.Remod/Add"] = "Yr.Renovated"
house_data_subset = subset(housing_data, 
                    select = c (SalePrice,
                                Lot.Area, 
                                Gr.Liv.Area,
                                Garage.Cars,
                                Total.Bsmt.SF,
                                Bedroom.AbvGr,
                                TotRms.AbvGrd,
                                Full.Bath, 
                                Half.Bath,
                                Overall.Qual,
                                Overall.Cond ,
                                Year.Built ,
                                Yr.Renovated,
                                Fireplaces,
                                Bsmt.Full.Bath,
                                Bsmt.Half.Bath,
                                Lot.Frontage,
                                Garage.Area,
                                MS.Zoning,
                                Bsmt.Qual,
                                Kitchen.Qual,
                                Paved.Drive,
                                Foundation,
                                Central.Air,
                                Garage.Type
                                ))
```

The following predictors have been identified as factor predictors.

```{r}
#Filter out and keep only Residential properties
target =  c("A (agr)", "C (all)", "I (all)")
house_data_subset = filter(house_data_subset, !MS.Zoning %in% target)
#Coerce as.factor for categorical variables
house_data_subset$Paved.Drive = as.factor(house_data_subset$Paved.Drive)
house_data_subset$MS.Zoning = as.factor(house_data_subset$MS.Zoning)
house_data_subset$Foundation = as.factor(house_data_subset$Foundation)
house_data_subset$Kitchen.Qual = as.factor(house_data_subset$Kitchen.Qual)
# add NA as factor
house_data_subset$Bsmt.Qual = factor(house_data_subset$Bsmt.Qual)
house_data_subset$Bsmt.Qual = addNA(house_data_subset$Bsmt.Qual, ifany = TRUE) 
house_data_subset$Garage.Type = factor(house_data_subset$Garage.Type)
house_data_subset$Garage.Type = addNA(house_data_subset$Garage.Type, ifany = TRUE)
```



## Exploratory Data Analysis

As a part of the exploratory data analysis,  various variables are pltted against SalePrice to assess their potential as useful predictors and to get an overall perspective for the dataset.

```{r , echo = FALSE, fig.height = 5, fig.width = 10}
par(mfrow=c(2,3))
plot(
  SalePrice ~ Lot.Area,
  data = housing_data,
  col = "dodgerblue",
  pch = 20,
  cex = 1.5,
  main = "House Sale Price, By Lot Area"
)
grid()
hist((house_data_subset$Gr.Liv.Area), 
     xlab   = "GR Living Area",
     breaks = 12,
     col = "dodgerblue",
     border = "darkorange",
     main = "Dist of House Sale Price Over Gr Liv Area")
grid()
boxplot(SalePrice ~ MS.Zoning, 
        data = house_data_subset, 
        xlab   = "MS Zoning",
        pch = 20, cex = 2, 
        col = "darkorange", 
        border = "dodgerblue",
        main = "Box plot Dist House Sale Price VS MS.Zoning")
grid()
YrBuild_Plot = as.data.frame(cbind(housing_data$SalePrice,housing_data$Year.Built,housing_data$Yr.Renovated,
                                   housing_data$Garage.Yr.Blt))
YrBuild_Plot = YrBuild_Plot[complete.cases(YrBuild_Plot),]
colnames(YrBuild_Plot) = c("SalePrice","Year.Built","Yr.Renovated","Garage.Yr.Blt")
plot(
  SalePrice ~ Year.Built ,
  data = YrBuild_Plot,
  col = "dodgerblue",
  pch = 20,
  cex = 1.5,
  main = "Sale Price, By Year Build"
)
grid()
plot(
  SalePrice ~ Yr.Renovated ,
  data = YrBuild_Plot,
  col = "dodgerblue",
  pch = 20,
  cex = 1.5,
  main = "Sale Price, By Year Remodeled"
)
grid()
plot(
  SalePrice ~ Garage.Yr.Blt ,
  data = YrBuild_Plot,
  col = "dodgerblue",
  pch = 20,
  cex = 1.5,
  main = "Sale Price, By Garage.Yr.Blt"
)
grid()
```


## Model Selection


### Predictor RelationShip

```{r}
Test_data = house_data_subset
form = as.formula(SalePrice ~ Fireplaces + Overall.Cond + Overall.Qual +
    Lot.Area + Bedroom.AbvGr + Year.Built + Gr.Liv.Area +
    Garage.Area )
mdl = rpart(form, data = Test_data)
rpart.plot(mdl, type = 5, clip.right.labs = FALSE, branch = .3, under = TRUE)
```

\

Below a pair plot like function is used, subselecting only numerical variables.

```{r}
chart.Correlation(house_data_subset[,1:10])
```

The correlation between numerical variables is examined, this highlights which variables will potentially be significant in the regression model.

```{r}
data = na.omit(house_data_subset)
corrplot(cor(data[sapply(data, is.numeric)]), method ="ellipse",
         title = "Correlation Matrix Graph",tl.cex = .5, tl.pos ="lt", tl.col ="dodgerblue" )
```

```{r}
house_correlation_var=cor(data.frame(house_data_subset[,1:18]), use = "complete.obs")
sort(abs(house_correlation_var["SalePrice", ]), decreasing = TRUE)
```

High levels of correlation among multiple predictors is confirmed by the above corrplot.

### Model Creation

We will now create a few models from the model selection process and perform Model diagnostics. We will check the Adjusted R2 for the models, and also check if any of the features have collinearity issues and high VIF. 



- *This is our first attempt to create a Full Additive model and a Stepwise AIC*

```{r}
full_model = lm(SalePrice ~ . , data = house_data_subset)
selected_model_aic = step(full_model, data = house_data_subset, trace = 0)
summary(selected_model_aic)$adj.r
```
This Model has an adjusted R2 of `r summary(selected_model_aic)$adj.r`


- *The analysis was started with a two way interactions model of all possible combinations, in order to identify which predictors makes more sense to us. From this assessment a couple of predictors were selected with a high p value. This analysis then utilizes those predictors as the base additive model.*

```{r}
FirstModel = lm(
  SalePrice ~ Fireplaces + TotRms.AbvGrd + Total.Bsmt.SF + Overall.Cond + Overall.Qual + Lot.Area + Kitchen.Qual + Bedroom.AbvGr + Year.Built + Full.Bath + Half.Bath + Gr.Liv.Area + Central.Air + Foundation + Garage.Area + Garage.Cars+ Garage.Type + Bsmt.Full.Bath  + Bsmt.Half.Bath,
  data = house_data_subset
)
```
This Model has adjusted R2 of `r summary(FirstModel)$adj.r`

- *Will use a Quadratic Relationship here*

```{r}
SecondModel_Quadratic = lm(SalePrice ~ Fireplaces + TotRms.AbvGrd + Total.Bsmt.SF + Overall.Cond + Overall.Qual + Lot.Area + Kitchen.Qual + Bedroom.AbvGr + Year.Built + Full.Bath + Half.Bath + Gr.Liv.Area + Central.Air + Foundation + Garage.Area + Garage.Cars + Garage.Type + Bsmt.Full.Bath  + Bsmt.Half.Bath + I(Fireplaces^2) + I(TotRms.AbvGrd^2) + I(Total.Bsmt.SF^ 2) + I(Overall.Cond^2) + I(Overall.Qual^2) +(Lot.Area^2) + I(Full.Bath^2) + I(Half.Bath^2) + I(Gr.Liv.Area^2) + I(Garage.Area^2)+ I(Garage.Cars^2) +I(Bsmt.Full.Bath^2) + I(Bsmt.Half.Bath^2) , data = house_data_subset)
```


- *AIC Backward selection for the Quadratic model*

```{r}
SecondModel = step(SecondModel_Quadratic,trace = 0)
```
This Model has adjusted R2 of `r summary(SecondModel)$adj.r`

- *Third model is based on Two Interaction relationships*

```{r}
ThirdModel = lm(SalePrice ~ (Fireplaces + TotRms.AbvGrd + Total.Bsmt.SF + Overall.Cond + Overall.Qual + Lot.Area + Kitchen.Qual + Bedroom.AbvGr + Year.Built + Full.Bath + Half.Bath + Gr.Liv.Area + Central.Air + Foundation + Garage.Area + Garage.Cars + Garage.Type + Bsmt.Full.Bath  + Bsmt.Half.Bath) ^ 2 , data = house_data_subset)
```
This Model has adjusted R2 of `r summary(ThirdModel)$adj.r`

- *One further model is checked by performing predictor and response transformation, also using an interaction term.*



```{r}
ForthModel = lm(
  log(SalePrice) ~ Fireplaces * Total.Bsmt.SF + Overall.Cond + Overall.Qual +
    log(Lot.Area) + Kitchen.Qual + Bedroom.AbvGr + Year.Built + log(Gr.Liv.Area) +
     Garage.Area + Garage.Cars, data = house_data_subset
)
summary(ForthModel)$adj.r
```
This Model has adjusted R2 of `r summary(ForthModel)$adj.r`


### Model Assumptions

The following analyzes the FirstModel, SecondModel , ThirdModel and ForthModel selection process.

```{r, echo = FALSE}
diagnostics = function(model, pcol = "dodgerblue", lcol = "darkorange", alpha = 0.01, plotit = TRUE, testit = TRUE){
  
    if(plotit == TRUE) {
    
    par(mfrow = c(1, 2))
    
    #Created fitted vs Residual Plot
    plot(fitted(model), resid(model), col = pcol, pch = 20, xlab = "Fitted", ylab = "Residuals", main = "Fitted versus Residuals")
    abline(h = 0, col = lcol, lwd = 2)
    grid()
    
    #Created QQ plot
    qqnorm(resid(model), main = "Normal Q-Q Plot", col = pcol)
    qqline(resid(model), col = lcol, lwd = 2)
    grid()
    
  }
  
  if(testit == TRUE) {
    
    #Derive p value and decision
    p_val = shapiro.test(resid(model))$p.value
    decision = ifelse(p_val < alpha, "Reject", "Fail to Reject")
    list(p_val = p_val, decision = decision)
    
  }
  
}
```

- **Linearity & Constant Variance**

```{r}
par(mfrow=c(4,2))
diagnostics(FirstModel, plotit = TRUE, testit = FALSE)
diagnostics(SecondModel, plotit = TRUE, testit = FALSE)
diagnostics(ThirdModel, plotit = TRUE, testit = FALSE)
diagnostics(ForthModel, plotit = TRUE, testit = FALSE)
```

- BP Test Results
```{r}
M1 = as.numeric(bptest(FirstModel)$p.value)
M2 = bptest(SecondModel)$p.value
M3 = bptest(ThirdModel)$p.value
M4 = bptest(ForthModel)$p.value
val= c(M1,M2,M3,M4)
output = data.frame(
  "ModelName" = c("`Model 1`", "`Model 2`", "`Model 3`", "`Model 4`"),"`BPTestPval`" = val
)
output
```

The above results shows that the P value is very low, hence homoscedasticity is rejected (the assumption of constant variance is not violated). In addition the BP value for the fourth model is better when compared to rest of the models.



- **Normailty of errors**


```{r, echo = FALSE, warning=FALSE,message=FALSE}
plot_qq = function(model, pointcol = "dodgerblue", linecol = "darkorange") {
  qqnorm(resid(model), col = pointcol, pch = 20, cex = 1.5)
  qqline(resid(model), col = linecol, lwd = 2)
}
par(mfrow=c(2,2))
hist(resid(FirstModel),xlab = "Residuals",,main = "Histogram of Residuals, firstModel",col = "darkorange",border = "dodgerblue")
#plot_qq(FirstModel)
hist(resid(SecondModel),xlab = "Residuals",,main = "Histogram of Residuals, secondModel",col = "darkorange",border = "dodgerblue")
#plot_qq(SecondModel)
hist(resid(ThirdModel),xlab = "Residuals",,main = "Histogram of Residuals, thirdModel",col = "darkorange",border = "dodgerblue")
#plot_qq(ThirdModel)
### Unusual Observations
hist(resid(ForthModel),xlab = "Residuals",,main = "Histogram of Residuals, fourthModel",col = "darkorange",border = "dodgerblue")
#plot_qq(ForthModel)
```

- Shapiro Wilk Test

```{r}
diagnostics(FirstModel, plotit = FALSE, testit = TRUE)
diagnostics(SecondModel, plotit = FALSE, testit = TRUE)
diagnostics(ThirdModel, plotit = FALSE, testit = TRUE)
diagnostics(ForthModel, plotit = FALSE, testit = TRUE)
```

The above Shapiro Test results shows that the P value is very low, hence we reject the null hypothesis (linearity is suspect). 

### Unusual Observations

- *Looking at Leverage, Outliers and Influential Data Points*

```{r}
#Leverage
lv_m1 = length(hatvalues(FirstModel)[hatvalues(FirstModel) > 2 * mean(hatvalues(FirstModel))])
lv_m2 = length(hatvalues(SecondModel)[hatvalues(SecondModel) > 2 * mean(hatvalues(SecondModel))])
lv_m3 = length(hatvalues(ThirdModel)[hatvalues(ThirdModel) > 2 * mean(hatvalues(ThirdModel))])
lv_m4 = length(hatvalues(ForthModel)[hatvalues(ForthModel) > 2 * mean(hatvalues(ForthModel))])
# Outliers
ol_m1 = length(rstandard(FirstModel)[abs(rstandard(FirstModel)) > 2])
ol_m2 = length(rstandard(SecondModel)[abs(rstandard(SecondModel)) > 2])
ol_m3 = length(rstandard(ThirdModel)[abs(rstandard(ThirdModel)) > 2])
ol_m4 = length(rstandard(ForthModel)[abs(rstandard(ForthModel)) > 2])
# Influential
inf_obs_m1 = length(cooks.distance(FirstModel)[cooks.distance(FirstModel) > 4 / length(cooks.distance(FirstModel))])
inf_obs_m2 = length(cooks.distance(SecondModel)[cooks.distance(SecondModel) > 4 / length(cooks.distance(SecondModel))])
inf_obs_m3 = length(cooks.distance(ThirdModel)[cooks.distance(ThirdModel) > 4 / length(cooks.distance(ThirdModel))])
inf_obs_m4 = length(cooks.distance(ForthModel)[cooks.distance(ForthModel) > 4 / length(cooks.distance(ForthModel))])
#Summarizing The Result
output = data.frame(
  "ModelName" = c("`Model 1`", "`Model 2`", "`Model 3`", "`Model 4`"),
  "TotalUnfluentialObs" = c(
    lv_m1 ,
    lv_m2,
    lv_m3,
    lv_m4
  ),
  "TotalOutliers" = c(
    ol_m1 ,
    ol_m2,
    ol_m3,
    ol_m3
  ),
  "InfluentialObs" = c(
    inf_obs_m1 ,
    inf_obs_m2,
    inf_obs_m3,
    inf_obs_m4
  )
  
)
knitr::kable(output)
```


###  Remove influential observation & Fix normality issue 

The Residual vs Residual plot and Normal QQ plot show that that they are not perfect and the test performed above also proves that there is something wrong with the models. and that a variance assumption is violated.

- Next a Box Cox Plot is performed on the model, after which transformations of Response and other variables are performed, to see if BP test values change.




```{r}
salepriceBCMod = caret::BoxCoxTrans(house_data_subset$SalePrice)
house_data_subset = cbind(house_data_subset, salePrice_new=predict(salepriceBCMod, house_data_subset$SalePrice)) 
LotAreaBCMod = caret::BoxCoxTrans(house_data_subset$Lot.Area)
house_data_subset = cbind(house_data_subset, LotArea_new=predict(LotAreaBCMod, house_data_subset$Lot.Area)) 
GrLivAreaBCMod = caret::BoxCoxTrans(house_data_subset$Gr.Liv.Area)
house_data_subset = cbind(house_data_subset, GrLivArea_new=predict(GrLivAreaBCMod, house_data_subset$Gr.Liv.Area)) 
```

- Final Model Stats

```{r}
cd_model4 = cooks.distance(ForthModel)
# Remove influential observation and use variable above created from Box Cox Plot
ForthModel_fixed = lm(
  salePrice_new ~ Fireplaces + Overall.Cond + Overall.Qual +
    LotArea_new + Bedroom.AbvGr + Year.Built + GrLivArea_new +
    Garage.Area ,
  data = house_data_subset,
  subset = cd_model4 <= 4 / length(cd_model4)
)
summary(ForthModel_fixed)$adj.r
bptest(ForthModel_fixed)
diagnostics(ForthModel_fixed, pcol = "grey", lcol = "green")
```

The 'FixedModel' looks very promising in terms of constant variance and linear relationship as compared to previous models.

```{r}
ncvTest(ForthModel_fixed)
spreadLevelPlot(ForthModel_fixed)
```

```{r}
boxcox(ForthModel_fixed, plotit = TRUE)
```


Based on the Results from box cox plot and ncvTest, it suggested us to do a power transformation of the Reponse Variable 

```{r}
cd_model4 = cooks.distance(ForthModel)
# Remove influential observation and use variable above created from Box Cox Plot
ForthModel_fixed = lm(
  (salePrice_new ^ -0.04237718) ~ Fireplaces + Overall.Cond + Overall.Qual +
    LotArea_new + Bedroom.AbvGr + Year.Built + GrLivArea_new +
    Garage.Area ,
  data = house_data_subset,
  subset = cd_model4 <= 4 / length(cd_model4)
)
summary(ForthModel_fixed)$adj.r
bptest(ForthModel_fixed)
diagnostics(ForthModel_fixed, pcol = "grey", lcol = "green")
```



## Evaluations


- *Reporting Adjusted $R2$ for each model*

```{r,warning=FALSE,message=FALSE}
output = data.frame(
  "ModelName" = c("`Model 1`", "`Model 2`", "`Model 3`", "`Model 4`", "Fixed Model"),
  "AdjustedR2" = c(
    summary(FirstModel)$adj.r.squared ,
    summary(SecondModel)$adj.r.squared,
    summary(ThirdModel)$adj.r.squared,
    summary(ForthModel)$adj.r.squared,
    summary(ForthModel_fixed)$adj.r.squared
  )
  
)
knitr::kable(output)
```

Fixed model is the leading model in terms of Adjusted R2.

- *Calculate `RMSE`  for each model*

```{r}
calc_loocv_rmse = function(model) {
sqrt(mean((resid(model) / (1 - hatvalues(model))) ^ 2))
}
output = data.frame(
  "ModelName" = c("`Model 1`", "`Model 2`", "`Model 3`", "`Model 4`", "`Fixed Model`"),
  "RMSE" = c(
    calc_loocv_rmse(FirstModel) ,
    calc_loocv_rmse(SecondModel),
    calc_loocv_rmse(ThirdModel),
   calc_loocv_rmse(ForthModel),
   calc_loocv_rmse(ForthModel_fixed)
  )
  
)
knitr::kable(output)
```
The fixed Model has a much better LOOCV RMSE, as compared to other models which show infinity.


- *Checking Variance Inflation for each of the model*
```{r}
# VIF
sort(vif(ForthModel_fixed))
sum(vif(FirstModel)>5)/length(coef(FirstModel))
sum(vif(SecondModel)>5)/length(coef(SecondModel))
sum(vif(ForthModel)>5)/length(coef(ForthModel))
sum(vif(ForthModel_fixed)>5)/length(coef(ForthModel_fixed))
```


- *AIC of all Models*
```{r}
output = data.frame(
  "ModelName" = c("`Model 1`", "`Model 2`", "`Model 3`", "`Model 4`", "`Fixed Model`"),
  "AIC" = c(AIC(FirstModel) ,
             AIC(SecondModel),
             AIC(ThirdModel),
             AIC(ForthModel),
             AIC(ForthModel_fixed)
             )
)
knitr::kable(output)
```


```{r, echo = FALSE}
rmse = function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}
get_rmse = function(model, data, response) {
  rmse(actual = data[, response], 
       predicted = predict(model, data))
}
get_complexity = function(model) {
  length(coef(model)) - 1
}
```

In addition to the above statistics, the model needs to be validated via fit and predictions on a training and testing set.

### Train VS Test Split for RMSE

```{r}
house_data_subset_mod = subset(house_data_subset,  cd_model4 <= 
    4/length(cd_model4))
```

```{r}
set.seed(108)
n = nrow(house_data_subset_mod)
house_data_subset_idx=sample(nrow(house_data_subset_mod),round(nrow(house_data_subset) / 2)) 
house_data_subset_trn = house_data_subset_mod[house_data_subset_idx, ]
house_data_subset_tst = house_data_subset_mod[-house_data_subset_idx, ]
```

- Recreated forth Model on Train Data
```{r}
ForthModel_fixed_train = lm(
  salePrice_new  ~ Fireplaces + Overall.Cond + Overall.Qual +
    LotArea_new + Bedroom.AbvGr + Year.Built + GrLivArea_new +
    Garage.Area ,
  data = house_data_subset_trn
  
)
```

RMSE for the train and test Model

```{r}
sqrt(mean((house_data_subset_trn$salePrice_new - predict(ForthModel_fixed_train, house_data_subset_trn)) ^ 2))
```


```{r}
sqrt(mean((house_data_subset_tst$salePrice_new - predict(ForthModel_fixed_train, house_data_subset_tst)) ^ 2))
```
The results delivered a train RMSE of 0.1148181 and a test RMSE 0.1159567. These are very close and do not seem to infer over or under ﬁtting. To further analyze ForthModel_ﬁxed, a forward and backward AIC and BIC is ran on ForthModel_ﬁxed. All four methods returned the same model. Calling this model ForthModel_AIC_BIC an anova test is conducted (between ForthModel_fixed and ForthModel_AIC_BIC) and the RMSEs from the training and the testing sets of both models are compared.

```{r}
ForthModel_AIC_BIC = lm(salePrice_new ~ Fireplaces + Overall.Cond + Overall.Qual + LotArea_new + Bedroom.AbvGr + Year.Built + GrLivArea_new,
  data = house_data_subset_trn)
```

```{r}
anova(ForthModel_fixed_train, ForthModel_AIC_BIC)
```

### ANOVA Test

The ANOVA test returns a very low p-value, consequently the larger model, aka the ForthModel_ﬁxed, is preferred.

```{r include=FALSE}
forth_trn = sqrt(mean((house_data_subset_trn$salePrice_new - predict(ForthModel_fixed_train, house_data_subset_trn)) ^ 2))
```


```{r include=FALSE}
forth_tst = sqrt(mean((house_data_subset_tst$salePrice_new - predict(ForthModel_fixed_train, house_data_subset_tst)) ^ 2))
```

```{r include=FALSE}
forthaic_trn = sqrt(mean((house_data_subset_trn$salePrice_new - predict(ForthModel_AIC_BIC, house_data_subset_trn)) ^ 2))
```


```{r include=FALSE}
forthaic_tst = sqrt(mean((house_data_subset_tst$salePrice_new - predict(ForthModel_AIC_BIC, house_data_subset_tst)) ^ 2))
```

### Final Output
```{r}
output = data.frame(
  "ModelName" = c("`ForthModel`", "`ForthModel_AIC_BIC`"),
  "Train_RMSE" = c(
     forth_trn,
    forthaic_trn
  ),
  "Test_RMSE" = c(
    forth_tst,
    forthaic_tst
  )
  
)
knitr::kable(output)
```




## Results & Discussion


Based on the results above, we conclude that it is not possible to make any further improvements to the model. Hence Final_Model is the ﬁnal and best model.

The following presents the final predictors:

```{r}
Final_Model = ForthModel_fixed
length(coef(Final_Model))
```

```{r}
names(coef(Final_Model))
```


```{r}
summary(Final_Model)
```


*The significant predictors.*

```{r}
summary(Final_Model)$coefficients[summary(Final_Model)$coefficients[ ,4] < 0.05,]
```

*Below is the accuracy of the model*

```{r,warning=FALSE,message=FALSE}
summary(Final_Model)$r.squared 
```


