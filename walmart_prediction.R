---
title: "project_2_initial"
output: html_document
---





```{r}
library(lubridate)
library(tidyverse)
```


```{r}
mypredict <- function(){
  
  # 1.  Remove any weekly sales data with na -> 0
  # There are ~15421 entries of this
  n.comp = 8
  train <- train %>%
    select(Store, Dept, Date, Weekly_Sales) %>%
    spread(Date, Weekly_Sales)
  train[is.na(train)] <- 0
  
  train_svd = NULL
  
  # 2. Run svd on training data and grab top n.comp (8) principal components
  # https://campuswire.com/c/G497EEF81/feed/654
  for(mydept in unique(train$Dept)){
    dept_data <- train %>% 
      filter(Dept == mydept)
    
    if (nrow(dept_data) > n.comp){
      tmp_data <- dept_data[, -c(1,2)]
      store_means <- rowMeans(tmp_data)
      tmp_data <- tmp_data - store_means
      z <- svd(tmp_data, nu=n.comp, nv=n.comp)
      s <- diag(z$d[1:n.comp])
      tmp_data <- z$u %*% s %*% t(z$v) + store_means
      tmp_data[tmp_data < 0] <- 0
      dept_data[, -c(1:2)] <- z$u %*% s %*% t(z$v) + store_means
    }
    train_svd = rbind(train_svd, dept_data)
  }
  train_svd <- train_svd %>% 
    gather(Date, Weekly_Sales, -Store, -Dept)
  
  # 2. Remove test data not in target window - reduces computation time of predictions
  # part of https://campuswire.com/c/G497EEF81/feed/490
  start_date <- ymd("2011-03-01") %m+% months(2 * (t - 1))
  end_date <- ymd("2011-05-01") %m+% months(2 * (t - 1))
  test_current <- test %>%
    filter(Date >= start_date & Date < end_date) %>%
    select(-IsHoliday)
  
  # 3. find the unique pairs of (Store, Dept) combo that appeared in both training and test sets
  # https://campuswire.com/c/G497EEF81/feed/632
  test_depts <- unique(test_current$Dept)
  train_pairs <- train_svd[, 1:2] %>% count(Store, Dept) %>% filter(n != 0)
  test_pairs <- test_current[, 1:2] %>% count(Store, Dept) %>% filter(n != 0)
  unique_pairs <- intersect(train_pairs[, 1:2], test_pairs[, 1:2])
  
  # pick out the needed training samples, convert to dummy coding, then put them into a list
  # https://campuswire.com/c/G497EEF81/feed/632
  
  # 4. Add Week and Year to features for prediction
  # https://campuswire.com/c/G497EEF81/feed/490
  train_split <- unique_pairs %>% 
    left_join(train_svd, by = c('Store', 'Dept')) %>% 
    # 5. Fix last week in year (b/c lubridate is weird) for 2010 (Dec 31 == 53 for some reason)
    # https://campuswire.com/c/G497EEF81/feed/490
    mutate(Wk = factor(ifelse(year(Date) == 2010, week(Date) - 1, week(Date)), levels = 1:52)) %>% 
    mutate(Yr = year(Date))
  train_split = as_tibble(model.matrix(~ Weekly_Sales + Store + Dept + Yr + Wk, train_split)) %>% group_split(Store, Dept)
  
  # do the same for the test set
  test_split <- unique_pairs %>% 
    left_join(test_current, by = c('Store', 'Dept')) %>% 
    mutate(Wk = factor(ifelse(year(Date) == 2010, week(Date) - 1, week(Date)), levels = 1:52)) %>% 
    mutate(Yr = year(Date))
  test_split = as_tibble(model.matrix(~ Store + Dept + Yr + Wk, test_split)) %>% mutate(Date = test_split$Date) %>% group_split(Store, Dept)
  
  # pre-allocate a list to store the predictions
  test_pred <- vector(mode = "list", length = nrow(unique_pairs))
  
  # 6. perform regression for each split, note we used lm.fit instead of lm
  for (i in 1:nrow(unique_pairs)) {
    tmp_train <- train_split[[i]]
    tmp_test <- test_split[[i]]
    
    mycoef <- lm.fit(as.matrix(tmp_train[, -(2:4)]), tmp_train$Weekly_Sales)$coefficients
    mycoef[is.na(mycoef)] <- 0
    tmp_pred <- mycoef[1] + as.matrix(tmp_test[, 4:55]) %*% mycoef[-1]
    
    test_pred[[i]] <- cbind(tmp_test[, 2:3], Date = tmp_test$Date, Weekly_Pred = tmp_pred[,1])
  }
  
  # turn the list into a table at once, 
  # this is much more efficient then keep concatenating small tables
  test_pred <- bind_rows(test_pred)
  
  return (test_pred)
} 

```




```{r}
# read raw data and extract date column
train_raw <- readr::read_csv(unz('train.csv.zip', 'train.csv'))
train_dates <- train_raw$Date
# training data from 2010-02 to 2011-02
start_date <- ymd("2010-02-01")
end_date <- start_date %m+% months(13)
# split dataset into training / testing
train_ids <- which(train_dates >= start_date & train_dates < end_date)
train = train_raw[train_ids, ]
test = train_raw[-train_ids, ]
# create the initial training data
readr::write_csv(train, 'train_ini.csv')
# create test.csv 
# removes weekly sales
test %>% 
  select(-Weekly_Sales) %>% 
  readr::write_csv('test.csv')
# create 10-fold time-series CV
num_folds <- 10
test_dates <- train_dates[-train_ids]
# month 1 --> 2011-03, and month 20 --> 2012-10.
# Fold 1 : month 1 & month 2, Fold 2 : month 3 & month 4 ...
for (i in 1:num_folds) {
    # filter fold for dates
    start_date <- ymd("2011-03-01") %m+% months(2 * (i - 1))
    end_date <- ymd("2011-05-01") %m+% months(2 * (i - 1))
    test_fold <- test %>%
        filter(Date >= start_date & Date < end_date)
    
    # write fold to a file
    readr::write_csv(test_fold, paste0('fold_', i, '.csv'))
}
```


```{r}
source("shyams4_mymain_08pm.R")
  
```

```{r}
```


```{r}
train <- readr::read_csv('train_ini.csv')
test <- readr::read_csv('test.csv')
num_folds <- 10
wae <- rep(0, num_folds)
for (t in 1:num_folds) {
  print(paste("Iteration eval:", t))
  # *** THIS IS YOUR PREDICTION FUNCTION ***
  test_pred <- mypredict()
  
  # load fold file 
  fold_file <- paste0('fold_', t, '.csv')
  new_train <- readr::read_csv(fold_file, 
                               col_types = cols())
  train <- train %>% add_row(new_train)
  # extract predictions matching up to the current fold
  scoring_tbl <- new_train %>% 
      left_join(test_pred, by = c('Date', 'Store', 'Dept'))
  
  # compute WMAE
  actuals <- scoring_tbl$Weekly_Sales
  preds <- scoring_tbl$Weekly_Pred
  preds[is.na(preds)] <- 0
  weights <- if_else(scoring_tbl$IsHoliday, 5, 1)
  wae[t] <- sum(weights * abs(actuals - preds)) / sum(weights)
}
print(wae)
mean(wae)
```

```{r}
# This is just so i can "run all above" quicker :)
```