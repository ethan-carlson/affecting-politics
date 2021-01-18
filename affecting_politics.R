setwd("Desktop/AC Final Analytics/") # change to your working directory

# install libraries
library(caret)
library(ggplot2)
library(ggcorrplot)
library(caTools)
#unknown
library(lars)
library(glmnet)
#XGBoost
library(xgboost)
library(plyr)
library(mlr)
library(ROCR)
library(DiagrammeR)
#Multilevel Linear Model
library(lme4)

set.seed(15071)


    ###Predict Affiliation from Affect Data###

data <- read.csv("regression_data_modified_clean.csv")

#convert all the the factor variables to be such
data$party = factor(data$party)
data$party = relevel(data$party, ref='Democrat')
data$ideology = factor(data$ideology)
data$candidate = factor(data$candidate)
data$condition = factor(data$condition)
data$condition = relevel(data$condition, ref='democrat')

data$disgust <- scale(data$disgust)
data$joy <- scale(data$joy)
data$valence <- scale(data$valence)
data$engagement <- scale(data$engagement)
data$polarity <- scale(data$polarity)
data$subjectivity <- scale(data$subjectivity)
data$rppg <- scale(data$rppg)

str(data) # check to make sure it worked

#check the correlation matrix
data_cor <- cor(data[,8:14])
ggcorrplot(data_cor)


#split the data to ensure that party is equally rep'd
trainIndex <- createDataPartition(data$party, p = .75,
                                  list = FALSE,
                                  times = 1)
train <- data[ trainIndex,]
test <- data[-trainIndex,]


#Multilevel Linear Regression
ml.lin.mod <- lmer(party ~ valence + (1 | userID),
                   data=train,
                   control=lmerControl(optimizer="Nelder_Mead",optCtrl=list(maxfun=1e5)))

summary(ml.lin.mod)

ml.lin.train.preds <- predict(ml.lin.mod, newdata = train)
ml.lin.test.preds <- predict(ml.lin.mod, newdata = test)

SSE = sum((train$like_trump - ml.lin.train.preds)^2)
SST = sum((train$like_trump - mean(train$like_trump))^2)
R2 = 1 - (SSE/SST)
SSE = sum((test$like_trump - ml.lin.test.preds)^2)
SST = sum((test$like_trump - mean(train$like_trump))^2)
OSR2 = 1 - (SSE/SST)
R2
OSR2


#CART

library(dplyr)
library(rpart)
library(rpart.plot)

cart1 = rpart(party ~ condition + disgust + joy + engagement + valence + polarity + subjectivity + rppg,
              data = train,
              parms=list(loss=cbind(c(0, 1), c(1, 0))),
              cp=0.01)
rpart.plot(cart1)
# make predictions on training set and see the in-sample confusion matrix
cart.train.preds <- predict(cart1, newdata = train, type = "class")
# make predictions on test set and see the OOS confusion matrix
cart.test.preds <- predict(cart1, newdata = test, type = "class")
table(test$party, cart.test.preds)

SSE = sum((train$party - cart.train.preds)^2)
SST = sum((train$party - mean(train$party))^2)
R2 = 1 - (SSE/SST)
SSE = sum((test$party - cart.test.preds)^2)
SST = sum((test$party - mean(train$party))^2)
OSR2 = 1 - (SSE/SST)
R2
OSR2

# plot the ROC
ROCRpred = prediction(cart.test.preds, test$party)
ROCCurve = performance(ROCRpred, "tpr", "fpr")
plot(ROCCurve, colorize=TRUE)
as.numeric(performance(ROCRpred, "auc")@y.values)


#split the data to ensure that party is equally rep'd
trainIndex <- createDataPartition(data$party, p = .7,
                                  list = FALSE,
                                  times = 1)
train <- data[ trainIndex,]
test <- data[-trainIndex,]

#logistic regression
lin.mod <- glm(formula = party ~ valence*condition, family = "binomial", data = train)
summary(lin.mod)

# make predictions on training set and see the in-sample confusion matrix
lin.train.preds <- predict(lin.mod, newdata = train)
# make predictions on test set and see the OOS confusion matrix
lin.test.preds <- predict(lin.mod, newdata = test)

# plot the ROC
ROCRpred = prediction(lin.test.preds, test$party)
ROCCurve = performance(ROCRpred, "tpr", "fpr")
plot(ROCCurve, colorize=TRUE)
as.numeric(performance(ROCRpred, "auc")@y.values)


#XGBoost CV
tg = expand.grid(max_depth = 1:5,
                 eta = c(0.1,0.2,0.3), 
                 subsample = 0.5, 
                 min_child_weight = 1, 
                 max_delta_step = c(0,1),
                 gamma = c(0.04,0.08,0.12),  
                 colsample_bytree = 1, 
                 alpha = c(0,1), 
                 lambda = c(0,1))
round.best = rep(0, nrow(tg))
RMSE.cv = rep(0, nrow(tg))
for(i in 1:nrow(tg))
{
  params.new = split(t(tg[i,]), colnames(tg))
  eval = xgb.cv(data = data.matrix(subset(train, select=-c(userID, party,ideology,candidate,like_joe,like_trump))), label= train$like_trump, params = params.new, nrounds = 100, nfold = 4)$evaluation_log$test_rmse_mean
  round.best[i] = which.min(eval)
  RMSE.cv[i] = eval[round.best[i]]
}
winner = which.min(RMSE.cv)

# print out winning parameters and round
tg[winner,]
round.best[winner]

# fill in winning params with hard-coded nums
params.winner = list(max_depth = 2,
                     eta = 0.1,
                     subsample = .5, 
                     min_child_weight = 1, 
                     max_delta_step = 0, 
                     gamma = 0.08,
                     colsample_bytree = 1, 
                     alpha = 0, 
                     lambda = 0)
round.winner = 20

# make the model with the relevant params
mod.xgboost <- xgboost(data = data.matrix(subset(train, select=-c(userID, party,ideology,candidate,like_joe,like_trump))), label= train$like_trump, params = params.winner, nrounds = round.winner, verbose = F)

# make predictions on training set and see the in-sample confusion matrix
xgb.train.preds <- predict(mod.xgboost, newdata = data.matrix(subset(train, select=-c(userID, party,ideology,candidate,like_joe,like_trump))))
# make predictions on test set and see the OOS confusion matrix
xgb.test.preds <- predict(mod.xgboost, newdata = data.matrix(subset(test, select=-c(userID, party,ideology,candidate,like_joe,like_trump))))

# plot the ROC
ROCRpred = prediction(xgb.test.preds, test$like_trump)
ROCCurve = performance(ROCRpred, "tpr", "fpr")
plot(ROCCurve, colorize=TRUE)
as.numeric(performance(ROCRpred, "auc")@y.values)

SSE = sum((train$like_trump - xgb.train.preds)^2)
SST = sum((train$like_trump - mean(train$like_trump))^2)
R2 = 1 - (SSE/SST)
SSE = sum((test$like_trump - xgb.test.preds)^2)
SST = sum((test$like_trump - mean(train$like_trump))^2)
OSR2 = 1 - (SSE/SST)
R2
OSR2

#feature importance
xgb.importance(model = mod.xgboost)


### LASSO ###

lasso <- expand.grid(alpha = 1,
                         lambda = 10^(seq(-5, 0, by = 0.1)))
cv_lasso <- caret::train(y = train$party,
                             x = data.matrix(subset(train, select=-c(userID, party,ideology,candidate,like_joe,like_trump))),
                             method = "glmnet",
                             trControl = trainControl(method="cv", number=5),
                             tuneGrid = lasso, 
                             family = "binomial")
cv_lasso$results
cv_lasso$bestTune

lasso.mod<- cv_lasso$finalModel
preds.l1 <- predict(lasso.mod, newx = data.matrix(subset(test, select=-c(userID, party,ideology,candidate,like_joe,like_trump))), s=cv_lasso$bestTune$lambda )

# plot the ROC to make sure we're happy
ROCRpred = prediction(preds.l1, test$party)
ROCCurve = performance(ROCRpred, "tpr", "fpr")
plot(ROCCurve, colorize=TRUE)
as.numeric(performance(ROCRpred, "auc")@y.values)

#feature importance
coef(lasso.mod, s = cv_lasso$bestTune$lambda)


    ###Predict Affect Data from Affiliation###

data <- read.csv("regression_data_modified_clean.csv")

#convert all the the factor variables to be such
data$party = factor(data$party)
data$party = relevel(data$party, ref='Democrat')
data$ideology = factor(data$ideology)
data$candidate = factor(data$candidate)
data$condition = factor(data$condition)
data$condition = relevel(data$condition, ref='democrat')
str(data) # check to make sure it worked

#split the data to ensure that Party is equally rep'd
trainIndex <- createDataPartition(data$party, p = .75,
                                  list = FALSE,
                                  times = 1)
train <- data[ trainIndex,]
test <- data[-trainIndex,]


#Multilevel Linear Regression
ml.lin.mod <- lmer(valence ~ party*condition + (1 | userID), data=train)
summary(ml.lin.mod)

ml.lin.train.preds <- predict(ml.lin.mod, newdata = train)
ml.lin.test.preds <- predict(ml.lin.mod, newdata = test)

SSE = sum((train$valence - ml.lin.train.preds)^2)
SST = sum((train$valence - mean(train$valence))^2)
R2 = 1 - (SSE/SST)
SSE = sum((test$valence - ml.lin.test.preds)^2)
SST = sum((test$valence - mean(train$valence))^2)
OSR2 = 1 - (SSE/SST)
R2
OSR2

#XGBoost CV
tg = expand.grid(max_depth = 1:5,
                 eta = c(0.05,0.1,0.15,0.2), 
                 subsample = 0.5, 
                 min_child_weight = 1, 
                 max_delta_step = 0,
                 gamma = c(0.05,0.1,0.15),  
                 colsample_bytree = 1, 
                 alpha = 0, 
                 lambda = 1)
round.best = rep(0, nrow(tg))
RMSE.cv = rep(0, nrow(tg))
for(i in 1:nrow(tg))
{
  params.new = split(t(tg[i,]), colnames(tg))
  eval = xgb.cv(data = data.matrix(subset(train, select=-c(userID, disgust))), label= train$disgust, params = params.new, nrounds = 100, nfold = 5)$evaluation_log$test_rmse_mean
  round.best[i] = which.min(eval)
  RMSE.cv[i] = eval[round.best[i]]
}
winner = which.min(RMSE.cv)

# print out winning parameters and round
tg[winner,]
round.best[winner]

# fill in winning params with hard-coded nums
params.winner = list(max_depth = 1, 
                     eta = 0.1, 
                     subsample = .5, 
                     min_child_weight = 1, 
                     max_delta_step = 0, 
                     gamma = 0.1, 
                     colsample_bytree = 1, 
                     alpha = 0, 
                     lambda = 1)
round.winner = 27

# make the model with the relevant params
mod.xgboost <- xgboost(data = data.matrix(subset(train, select=-c(userID, disgust))), label= train$disgust, params = params.winner, nrounds = round.winner, verbose = F)

# make predictions on training set and see the in-sample confusion matri
xgb.train.preds <- predict(mod.xgboost, newdata = data.matrix(subset(train, select=-c(userID, disgust))))

# make predictions on test set and see the OOS confusion matrix
xgb.test.preds <- predict(mod.xgboost, newdata = data.matrix(subset(test, select=-c(userID, disgust))))

# calculate R2 and OSR2
SSE = sum((train$disgust - xgb.train.preds)^2)
SST = sum((train$disgust - mean(train$disgust))^2)
R2 = 1 - (SSE/SST)
SSE = sum((test$disgust - xgb.test.preds)^2)
SST = sum((test$disgust - mean(train$disgust))^2)
OSR2 = 1 - (SSE/SST)
R2
OSR2

#feature importance
xgb.importance(model = mod.xgboost)


### LASSO ###

idm_lasso <- expand.grid(alpha = 1,
                         lambda = 10^(seq(-2, 10, by = 0.1)))
#lambda = lasso.lambda.opt)
idm_cv_lasso <- caret::train(y = train$subjectivity,
                             x = data.matrix(subset(train, select=-c(userID, subjectivity))),
                             method = "glmnet",
                             trControl = trainControl(method="cv", number=5),
                             tuneGrid = idm_lasso)
idm_cv_lasso$results
idm_cv_lasso$bestTune

lasso.mod<- idm_cv_lasso$finalModel
train.preds.l1 <- predict(lasso.mod, newx = data.matrix(subset(train, select=-c(userID, subjectivity))), s=idm_cv_lasso$bestTune$lambda )
test.preds.l1 <- predict(lasso.mod, newx = data.matrix(subset(test, select=-c(userID, subjectivity))), s=idm_cv_lasso$bestTune$lambda )

# calculate R2 and OSR2
SSE = sum((train$subjectivity - train.preds.l1)^2)
SST = sum((train$subjectivity - mean(data$subjectivity))^2)
R2 = 1 - (SSE/SST)
SSE = sum((test$subjectivity - test.preds.l1)^2)
SST = sum((test$subjectivity - mean(data$subjectivity))^2)
OSR2 = 1 - (SSE/SST)
R2
OSR2

#feature importance
coef(lasso.mod, s = idm_cv_lasso$bestTune$lambda)


#CART

library(dplyr)
library(rpart)
library(rpart.plot)

cpVals <- data.frame(.cp = seq(0.01, .03, by=.001))
Loss <- function(data, lev = NULL, model = NULL, ...) {c(AvgLoss = mean(sum((data$obs - data$pred)^2))) }
trControl <- trainControl(method="cv", number=10, summaryFunction=Loss)
train.cart <- caret::train(train %>% select(-c(userID,subjectivity)), train$subjectivity, trControl=trControl, method="rpart", tuneGrid=cpVals, metric="AvgLoss", maximize=FALSE)
mod.cart <- train.cart$finalModel
mod.cart$cp

