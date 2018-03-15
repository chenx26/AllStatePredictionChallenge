rm(list=ls())
library(Matrix)
library(data.table)		
library(xgboost)      # Extreme Gradient Boosting
library(mlr)


ids_test = fread("./data/test.csv")
# data_all = data_all[1:10000]
# str(data_train_merged)

train_data = read.csv("./data/dtrain.csv", header = TRUE)[-1]
last_event = factor(read.csv("./data/last_event.csv", header = TRUE)$last_event)
train_data = cbind(train_data, last_event)
unique_events = sort(unique(unlist(apply(train_data, 2, function(x) unique(x)))))
for (i in 1:ncol(train_data)){
  train_data[,i] = factor(train_data[,i], levels = unique_events)
}
# train_data = cbind(train_data, dummy = 0.0)

# train_data = sparse.model.matrix(last_event ~ . - 1, data = train_data)
train_data_ohe = model.matrix(last_event ~ . - 1, data = train_data)
str(train_data)
colnames(train_data)
# prepare data for training the model

train_idx = sample(1:nrow(train_data_ohe), nrow(train_data_ohe) * 0.75)
dtrain = train_data_ohe[train_idx,]
dtrain[1,1] = as.numeric(dtrain[1,1])
dtest = train_data_ohe[-train_idx,]
dtest[1,1] = as.numeric(dtest[1,1])
#dtrain = xgb.DMatrix(data = dtrain, label = last_event[train_idx])
#dtest = xgb.DMatrix(data = dtest, label = last_event[-train_idx])

### Start of algorithm

#default parameters
params <- list(
  booster = "gbtree",
  objective = "multi:softprob",
  eta=0.3,
  gamma=0,
  max_depth=6,
  min_child_weight=1,
  subsample=1,
  colsample_bytree=1,
  num_class = length(unique_events),
  eval_metric = "mlogloss"
)

#set parallel backend
library(doMC)
registerDoMC(cores = 4)

xgbcv <- xgb.cv(params = params
                ,data = dtrain
                ,label = last_event[train_idx]
                ,nrounds = 200
                ,nfold = 5
                ,showsd = T
                ,stratified = T
                ,print.every.n = 10
                ,early.stop.round = 20
                ,maximize = F
)

xgb1 = xgboost(data = dtrain
               ,label = last_event[train_idx]
               ,params = params
               ,nrounds = 140
               ,print.every.n = 10
               ,early.stop.round = 10
               ,maximize = F)

#first default - model training
# xgb1 <- xgb.train(
#   params = params
#   ,data = dtrain
#   ,label = last_event[train_idx]
#   ,nrounds = xgbcv$best_iteration
#   ,watchlist = list(val=dtest,train=dtrain)
#   ,print.every.n = 10
#   ,early.stop.round = 10
#   ,maximize = F
# )

#model prediction
numberOfClasses = length(unique_events)
test_pred <- predict(xgb1, dtest)
y_pred = matrix(test_pred, nrow = nrow(dtest), byrow = TRUE)[,-1]
MLmetrics::MultiLogLoss(y_true = last_event[-train_idx], y_pred = y_pred)

### train model on all training data

xgb_all = xgboost(data = train_data_ohe
               ,label = last_event
               ,params = params
               ,nrounds = 50
               ,print.every.n = 10
               ,early.stop.round = 10
               ,maximize = F)

# xgb_all <- xgb.train(
#   params = params
#   ,data = xgb.DMatrix(data = train_data, label = data_train_merged$last_event)
#   ,nrounds = xgbcv$best_iteration
#   ,watchlist = list(val=dtest,train=dtrain)
#   ,print.every.n = 10
# #  ,early.stop.round = 10
#   ,maximize = F
# )


##### Scoring the model on the test data

# create the test dataset
test_all = read.csv("./data/dtest.csv", header = TRUE)
test_all$Event24[1] = 30021
test_all$Event25[1] = 30021
for (i in 1:ncol(test_all)){
  test_all[,i] = factor(test_all[,i])
}

dtest_all = sparse.model.matrix(id ~ . - 1, data = test_all)


test_all.pred = predict(xgb_all, dtest_all)
test_all.pred.mat = matrix(test_all.pred, nrow = nrow(dtest_all), byrow = TRUE)[,-1]
# testtask_all <- makeClassifTask(data = test_all,target = "last_event")
str(test_all.pred)
predictions = as.data.table(test_all.pred.mat)
setnames(predictions, names(predictions), as.character(unique_events)[-1])
predictions[, id:=ids_test$id]
str(predictions)
pred_columns <- colnames(predictions)
pred_columns <- c('id', sort(pred_columns[-length(pred_columns)]))
setcolorder(predictions, pred_columns)
setnames(predictions, c(pred_columns[1], 
                        paste('event_', pred_columns[-1], sep='')))


write.csv(predictions, file="./data/XGBoost_no_mlr_all_features.csv", quote=TRUE, row.names=FALSE)


parallelStop()

