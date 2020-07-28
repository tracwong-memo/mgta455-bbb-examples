## Using nnet and caret to analyses the bbb data
## loading libraries
library(nnet)
library(caret)
library(radiant)

## some formating options
options(
  width = 250,
  scipen = 100,
  max.print = 5000,
  stringsAsFactors = FALSE
)

## loading the bbb data
bbb <- readr::read_rds("data/bbb.rds")

## response (rvar) and explanatory (evar) variables
rvar <- "buyer"
evar <- c("gender", "last", "total", "child", "youth", "cook", "do_it", "reference", "art", "geog")
lev <- "yes"

## demo of onehot encoder function
oh <- select_at(bbb, c(rvar, evar)) %>% 
  mutate(buyer = factor(buyer, levels = c("no", "yes"))) %>% 
  onehot()
head(oh[, -1])

## creating a new training variable
split <- randomizer(select(bbb, -training), vars = "", conditions = 0:1, blocks = "buyer", probs = c(0.3, 0.7), label = "training")
summary(split)
head(split$dataset)
bbb <- split$dataset
training <- bbb[["training"]]
ind <- training == 1

## create a training and a validation (or test) set
## for caret to work with
df_train <- bbb[ind, c(rvar, evar)]
df_test <- bbb[-ind, c(rvar, evar)]

## setup a tibble to use for evaluation
eval_dat <- tibble::tibble(
  buyer = bbb$buyer,
  training = training
)

## compare to a logistic regression
result <- logistic(df_train,  rvar = rvar,  evar = evar,  lev = lev)
summary(result)
eval_dat$logit <- predict(result, bbb)$Prediction

## using radiant.model::nn
## note that radiant.model::nn automatically applies variable scaling
result <- nn(df_train, rvar = rvar, evar = evar, lev = lev, size = 3, decay = 0.15, seed = 1234)
summary(result)
eval_dat$nn3r <- predict(result, bbb)$Prediction

## using the radiant.model::cv.nn function for cross validation
## use the big grid ...
# cv.nn(result, K = 10, size = 1:6, decay = seq(0, 1, 0.05))
## ... or the small grid as an example
cv.nn(result, K = 5, size = 1:3, decay = seq(0.05, 0.15, 0.05), seed = 1234)

## standardize data for use with the nnet package
## numeric and integer variables will have mean = 0 and sd = 0.5
## change sf to 1 for mean = 0 and sd = 1
df_train_scaled <- scale_df(df_train, sf = 2)
str(df_train_scaled)

## check that numeric and integer variables have mean = 0 and sd = 0.5
summarize_if(df_train_scaled, is.numeric, list(mean, sd)) %>%
  gather() %>%
  format_df()

## scale the bbb data using the mean and standard deviation of the
## training sample (see ?radiant.model::scaledf)
bbb_scaled <- bbb %>%
  copy_attr(df_train_scaled, c("radiant_ms","radiant_sds")) %>%
  scale_df(calc = FALSE)

str(bbb_scaled)

## running an nnet model with 2 nodes in the hidden layer
## will produce the same results as the nn model above
set.seed(1234)
result <- nnet::nnet(
  buyer == lev ~ .,
  data = df_train_scaled,
  size = 3,
  decay = 0.15,
  rang = .1,
  linout = FALSE,
  entropy = TRUE,
  skip = FALSE,
  trace = FALSE,
  maxit = 10000
)
eval_dat$nn3 <- predict(result, bbb_scaled)[,1]

## check that the radiant (nnr) and nnet (nn2) predictions are the same
head(eval_dat)

## Example of using a custom function to evaluate model
my_auc <- function(data, lev = NULL, model = NULL) {
  c(my_auc = radiant.model::auc(data[[lev[1]]], data$obs, lev[1]))
}

## using caret with nnet
## use the big grid ...
# grid <- expand.grid(size = 1:6, decay = seq(0, 1, 0.05))
## ... or the small grid as an example
grid <- expand.grid(size = 3, decay = seq(0.05, 0.15, 0.05))
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = my_auc,
  verboseIter = TRUE
)

## this can take quite some time, especially with a big grid and model ...
## comment out to avoid running is not needed
set.seed(1234)
result <- train(
  select(df_train_scaled, -1),
  df_train_scaled[[rvar]],
  method = "nnet",
  trControl = ctrl,
  tuneGrid = grid,
  metric = "my_auc",
  rang = .1,
  skip = FALSE,
  linout = FALSE,
  trace = FALSE,
  maxit = 10000
)

##  Running through the above code with the small grid and cv = 5 produces ...
result$results
# Aggregating results
# Selecting tuning parameters
# Fitting size = 3, decay = 0.05 on full training set

## extract the uncovered tuning parameters
tuned <- result$bestTune

## re-running final model so we can control the seed
## and make sure the results are reproducible
set.seed(1234)
result <- nnet::nnet(
  buyer == lev ~ .,
  data = df_train_scaled,
  size = tuned$size,
  decay = tuned$decay,
  rang = .1,
  linout = FALSE,
  entropy = TRUE,
  skip = FALSE,
  trace = FALSE,
  maxit = 10000
)
eval_dat$nnc <- predict(result, bbb_scaled)[,1]

## get a list of the models to compare
mods <- colnames(eval_dat)[-1:-2]

## evaluate all models using the validation dataset
evalbin(
  eval_dat,
  pred = mods,
  rvar = rvar,
  lev = lev,
  qnt = 50,
  train = "Test",
  data_filter = "training == 1"
) %>% plot(plots = "gains")

## evaluate the model picked by caret in training and validation
## for evidence of overfitting in the validation (or test) data
evalbin(
  eval_dat,
  pred = "nnc",
  rvar = rvar,
  lev = lev,
  qnt = 50,
  train = "Both",
  data_filter = "training == 1"
) %>% plot(plots = "gains")

## calculate the confusion matrix and various performance
## metrics for all models
confusion(
  eval_dat,
  pred = mods,
  rvar = rvar,
  lev = lev,
  qnt = 50,
  train = "Test",
  data_filter = "training == 1"
) %>% summary()

## Example below how to use use caret and nn for regression
## of `total` on the other variables in bbb data
## Note: This is NOT meant to be a meaningful regression, just an example
grid <- expand.grid(size = 1:2, decay = seq(0.25, 0.5, 0.25))
ctrl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)

## this will take some time ...
set.seed(1234)
result <- train(
  select_at(df_train_scaled, setdiff(c(rvar, evar), "total")),
  df_train_scaled$total,
  method = "nnet",
  trControl = ctrl,
  tuneGrid = grid,
  linout = TRUE,
  entropy = FALSE,
  skip = FALSE,
  rang = .1,
  trace = FALSE,
  maxit = 10000
)

##  Running through the above code produces ...
# Aggregating results
# Selecting tuning parameters
# Fitting size = 1, decay = 0.25 on full training set
result$results

## extract the uncovered tuning parameters
tuned <- result$bestTune

## re-estimating the final model so we can control the seed
set.seed(1234)
result <- nnet::nnet(
  total ~ .,
  data = df_train_scaled,
  size = tuned$size,
  decay = tuned$decay,
  rang = .1,
  linout = TRUE,
  entropy = FALSE,
  skip = FALSE,
  trace = FALSE,
  maxit = 10000
)

## getting predictions back on the original scale
total_mean <- attributes(bbb_scaled)$radiant_ms$total
total_sd <- attributes(bbb_scaled)$radiant_sds$total
pred <- predict(result, bbb_scaled, type = "raw") * 2 * total_sd + total_mean
head(pred[,1])

## estimating the same model using radiant.model::nn
## note that radiant.model::nn will scale and re-scale
## the data automatically for you
result <- nn(
  df_train,
  rvar = "total",
  evar = setdiff(c(rvar, evar), "total"),
  type = "regression",
  size = tuned$size,
  decay = tuned$decay,
  seed = 1234
)
summary(result, prn = TRUE)
pred <- predict(result, bbb)$Prediction
head(pred)
