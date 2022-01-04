---
title: "p1"
output: html_document
---

# Use http://jse.amstat.org/v19n3/decock/DataDocumentation.txt
# for ref on variables/column info

```{r}
library(glmnet)
library(randomForest)
library(xgboost)
library(dplyr)
library(tidyverse)
```

# Preprocessing

Right now we are just loading in one singular training/test split and working from that

```{r}
train <- read.csv("train.csv", stringsAsFactors = TRUE)
test <- read.csv("test.csv", stringsAsFactors = TRUE)
test.y <- read.csv("test_y.csv", stringsAsFactors = TRUE)
# There are 108, 51 total na (train, test respectively), each being in the Garage_Yr_Blt column
# b/c no garage was built
# https://campuswire.com/c/G497EEF81/feed/379
# Replace na with 0 for Garage YR Blt
train[is.na(train)] = 0
test[is.na(test)] = 0
```

```{r}
# Remove suggested variables to start
# https://campuswire.com/c/G497EEF81/feed/420
remove.var <- c('Street', 'Utilities', 'Alley','Condition_2', 'Roof_Matl', 'Heating', 'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude','Latitude')
train = train[,  !names(train) %in% remove.var]
test = test[,  !names(test) %in% remove.var]
```

```{r}
# Remove BsmtFin_SF_1 (Type 1 finished square feet) b.c the values just dont make sense
# There are hundreds of data points but they only take on the values of 
# 2 6 1 3 7 4 5 0 ... 
remove.var <- c('BsmtFin_SF_1')
train = train[,  !names(train) %in% remove.var]
test = test[,  !names(test) %in% remove.var]
```

```{r}
impute_tst_with_trn = function(trn, tst, how=c("median", "most")){
  # INpute either via most frequent or most avg frequency
  if(how == "median"){
    ordered_freq = sort(table(trn))
    avg_freq_level = names(ordered_freq[floor(length(ordered_freq)/2)])
    tst[is.na(tst)] = avg_freq_level
  }
  else {
    most_freq_level = names(which.max(table(trn)))
    tst[is.na(tst)] = most_freq_level
  }
  return(tst)
}
```


```{r}
# This is a start for fixing levels
# But for example the train set has 9 levels for Sale Type
# and the test set has 10
# A value of NA is set to the ones in that extra level
# Here is a recommendation from the prof how to handle factors
# https://campuswire.com/c/G497EEF81/feed/397
adjust.levels.to.train.set <- function(trn, tst){
  if(is.factor(trn)){
    
    new_tst = factor(tst, levels=c(levels(trn)))
    # For any missing values due to mismatched levels
    # they get sent to NA, replace them with the most frequent level inside the train set
    new_tst = impute_tst_with_trn(trn, new_tst, "median")
    return(new_tst)
  }
  else{
    return (tst)
  }
}
for(var in colnames(train)){
  if(var %in% colnames(test)){
    test[,var] = adjust.levels.to.train.set(train[,var], test[,var])
  }
}
```


```{r}
# Get rid of PID in training b/c that is just essentially an item index
# Professor doesnt say anything about PID, maybe add it in - but it doesnt make sense to
train.x = train[, !names(train) %in% c("PID", "Sale_Price")]
test.x = test[, !names(test) %in% c("PID", "Sale_Price")]
train.y = train[,"Sale_Price"]
train.y.log = log(train[,"Sale_Price"])
train.x$Garage_Yr_Blt[is.na(train.x$Garage_Yr_Blt)] = 0
```


```{r}
# Window limit (winsorize) some params with outliers or params that have diminishing returns
# https://campuswire.com/c/G497EEF81/feed/420
winsor.vars <- c("Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF", "Second_Flr_SF", 'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF", "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val")
quan.value <- 0.95
for(var in winsor.vars){
  tmp <- train.x[, var]
  myquan <- quantile(tmp, probs = quan.value, na.rm = TRUE)
  tmp[tmp > myquan] <- myquan
  train.x[, var] <- tmp
}
# Best approach would be to use quantiles from training data on test data
# but this is acceptable for now
for(var in winsor.vars){
  tmp <- test.x[, var]
  myquan <- quantile(tmp, probs = quan.value, na.rm = TRUE)
  tmp[tmp > myquan] <- myquan
  test.x[, var] <- tmp
}
```

```{r}
# Last processing step
# This converts each factor variable column to multiple dummy variable columns
# so if we have a column with data ["one","two","three","one"] with levels "one","two","three"
# it will make 3 columns, with basically one-hot encodings
# such as col1: [1,0,0,1], col2: [0,1,0,0], col3: [0,0,1,0]
x_train <- model.matrix( ~ .-1, train.x)
x_test <- model.matrix( ~ .-1, test.x)
```


# Model work


```{r}
# Loop over various alpha values to find best one for us
# Plot results after
alpha_vals = seq(0, 1, by=0.05)
n = length(alpha_vals)
rsmes = matrix(0,nrow=n, ncol=1)
for(i in 1:n){
  cv.out <- cv.glmnet(x_train, train.y.log, alpha = alpha_vals[i])
  test.y.pred <-predict(cv.out, s = cv.out$lambda.min, newx = x_test)
  # Only check preds  against sale price, not PID
  rsmes[i]= sqrt(mean(((test.y.pred) - log(test.y$Sale_Price))^2))
}
plot(alpha_vals, rsmes)
```
```{r}
cv.out <- cv.glmnet(x_train, train.y.log, alpha = 1)
test.y.pred <-predict(cv.out, s = cv.out$lambda.min, newx = x_test)
```

```{r}
# Concatenate PID and predicted Sales price
final.out.y = exp(test.y.pred)
output = cbind(test[,1], final.out.y)
colnames(output) = c("PID", "Sale_Price")
write.csv(output, 'mysubmission1.txt', row.names=FALSE)
```

```{r}
# Check the format works for evaluation
pred <- read.csv("mysubmission1.txt")
test.y <- read.csv("test_y.csv", stringsAsFactors = TRUE)
new.test.y = test.y
names(new.test.y)[2] <- "True_Sale_Price"
pred <- merge(pred, new.test.y, by="PID")
sqrt(mean((log(pred$Sale_Price) - log(pred$True_Sale_Price))^2))
```


# XgBoost Implementation

```{r}
categorical.vars <- colnames(train.x)[
  which(sapply(train.x, 
               function(x) is.factor(x)))]
# Remove categorical vars for now - add them back in later
train.matrix <- train.x[, !colnames(train.x) %in% categorical.vars, 
                          drop=FALSE]
n.train <- nrow(train.matrix)
for(var in categorical.vars){
    mylevels <- sort(unique(train.x[, var]))
    m <- length(mylevels)
    m <- ifelse(m>2, m, 1)
    tmp.train <- matrix(0, n.train, m)
    col.names <- NULL
    # Loop over each level
    # Make 1 hot encodings based on original date
    # Saved into new tmp.train matrix
    for(j in 1:m){
      tmp.train[train.x[, var]==mylevels[j], j] <- 1
      # Incrementally add all level names to a list
      col.names <- c(col.names, paste(var, '_', mylevels[j], sep=''))
      }
    colnames(tmp.train) <- col.names
    train.matrix <- cbind(train.matrix, tmp.train)
  }
```

```{r}
set.seed(5290)
xgb.model <- xgboost(data = as.matrix(train.matrix), 
                      # Do prediction on log sale price as better results
                       label = log(train.y), max_depth = 6,
                       eta = 0.05, nrounds = 5000,
                       subsample = 0.5,
                       verbose = FALSE)
```

```{r}
categorical.vars <- colnames(test.x)[
  which(sapply(test.x, 
               function(x) is.factor(x)))]
test.matrix <- test.x[, !colnames(test.x) %in% categorical.vars, 
                          drop=FALSE]
n.test <- nrow(test.matrix)
for(var in categorical.vars){
    mylevels <- sort(unique(test.x[, var]))
    m <- length(mylevels)
    m <- ifelse(m>2, m, 1)
    tmp.test <- matrix(0, n.test, m)
    col.names <- NULL
    for(j in 1:m){
      tmp.test[test.x[, var]==mylevels[j], j] <- 1
      col.names <- c(col.names, paste(var, '_', mylevels[j], sep=''))
      }
    colnames(tmp.test) <- col.names
    test.matrix <- cbind(test.matrix, tmp.test)
  }
```

```{r}
isect = c(intersect(colnames(train.matrix), colnames(test.matrix)))
c_add = setdiff(colnames(train.matrix), isect)
test.matrix[,c_add] <-0
c_del = setdiff(colnames(test.matrix), colnames(train.matrix))
test.matrix = test.matrix[,! colnames(test.matrix) %in% c_del]
```

```{r}
xgb_pred = predict(xgb.model, newdata = as.matrix(test.matrix[,xgb.model$feature_names]))
```

```{r}
sqrt(mean(((xgb_pred) - log(test.y$Sale_Price))^2))
```
```{r}
# Concatenate PID and predicted Sales price
final.out.y = exp(xgb_pred)
output = cbind(test[,1], final.out.y)
colnames(output) = c("PID", "Sale_Price")
write.csv(output, 'mysubmission2.txt', row.names=FALSE)
```

```{r}
# Check the format works for evaluation
pred <- read.csv("mysubmission2.txt")
test.y <- read.csv("test_y.csv", stringsAsFactors = TRUE)
new.test.y = test.y
names(new.test.y)[2] <- "True_Sale_Price"
pred <- merge(pred, new.test.y, by="PID")
sqrt(mean((log(pred$Sale_Price) - log(pred$True_Sale_Price))^2))
```
