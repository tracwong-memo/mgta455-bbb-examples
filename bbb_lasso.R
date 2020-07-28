library(glmnet)
# library(glmnetUtils)
library(radiant)
library(caret)

## some formating options
options(
  width = 250,
  scipen = 100,
  max.print = 5000,
  stringsAsFactors = FALSE
)

## loading the bbb data
# adding a random number that should be dropped by lasso
bbb <- readr::read_rds("data/bbb.rds") %>%
  mutate(rnd = rnorm(n()))

## response (rvar) and explanatory (evar) variables
rvar <- "buyer"
evar <- c("gender", "last", "total", "child", "youth", "cook", "do_it", "reference", "art", "geog", "rnd")
lev <- "yes"
form <- as.formula(glue('{rvar} ~ {glue_collapse(evar, " + ")}'))

## creating a new training variable
split <- randomizer(select(bbb, -training), vars = "", conditions = 0:1, blocks = "buyer", probs = c(0.3, 0.7), label = "training")
summary(split)
head(split$dataset)
bbb <- split$dataset

# the logistic function in Radiant handles standardization
lr <- logistic(bbb, form = form, lev = "yes", check = "standardize", data_filter = "training == 1")

# using glmnet to create a "smartly" chosen grid
mod <- glmnet(model.matrix(lr$model)[, -1], lr$model$y, alpha = 1, family = "binomial", lambda = 0.003)
coef(mod)
mod <- glmnet(model.matrix(lr$model)[, -1], lr$model$y, alpha = 1, family = "binomial")
lambda_grid <- mod$lambda

ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  allowParallel = TRUE,                 # This tell caret to use paralell processing
  summaryFunction = twoClassSummary,   # Tells caret to compute AUC (ROC)
  verboseIter = TRUE
)

fit_data <- lr$model$model
fit_data[[rvar]] <- filter(bbb, training == 1) %>% pull(rvar)

glmnetFit <- train(
  form, 
  data = fit_data,
  method = "glmnet",
  family = "binomial",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = data.frame(alpha = 1, lambda = lambda_grid)
)

plot(glmnetFit)
glmnetFit$bestTune
coef(glmnetFit$finalModel, glmnetFit$finalModel$lambdaOpt)

# NO variables are removed so we can estimate and predict for the 
# training and test set as usual 
