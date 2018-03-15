rm(list=ls())
library(Matrix)
library(data.table)		
library(xgboost)      # Extreme Gradient Boosting
library(mlr)

# read all data
# for the train set, the number of events go from 2 to 25, excluding the last event
# for the test set, the number of events go from 2 to 23

train_number_of_features = 25
test_number_of_features = 23


### fit xgboost tree with parameter tuning

train_data = read.csv("./data/dtrain.csv", header = TRUE)[-1]
last_event = read.csv("./data/last_event.csv", header = TRUE)
train_data = cbind(train_data, last_event)
unique_events = sort(unique(unlist(apply(train_data, 2, function(x) unique(x)))))
for (i in 1:ncol(train_data)){
  train_data[,i] = factor(train_data[,i])
}
train_data = cbind(train_data, dummy = 0.0)
train_idx = sample(1:nrow(train_data), nrow(train_data) * 0.75)
train = train_data[train_idx,]
test = train_data[-train_idx,]
traintask <- makeClassifTask(data = train,target = "last_event")
testtask <- makeClassifTask(data = test,target = "last_event")
lrn <- makeLearner("classif.xgboost",predict.type = "prob")
lrn$par.vals <- list(
  objective="multi:softprob",
  eval_metric="mlogloss",
  nrounds=100,
  booster="gbtree",
#  eta=0.1,
  num_class = length(unique_events)
)

params <- makeParamSet(
#  makeDiscreteParam("booster",values = c("gbtree","gblinear")),
  makeIntegerParam("max_depth",lower = 1L,upper = 1L),
  makeNumericParam("min_child_weight",lower = 1L,upper = 1L),
  makeNumericParam("subsample",lower = 1,upper = 1),
  makeNumericParam("colsample_bytree",lower = 1,upper = 1),
  makeNumericParam("eta",lower = 0.1,upper = 0.1)
)

rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)
ctrl <- makeTuneControlRandom(maxit = 10L)

#set parallel backend
library(parallel)
library(parallelMap)
parallelStartSocket(cpus = 4)

#parameter tuning
mytune <- tuneParams(learner = lrn
                     ,task = traintask
                     ,resampling = rdesc
                     ,measures = logloss
                     ,par.set = params
                     ,control = ctrl
                     ,show.info = T)

mytune$y
mytune$x

lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)

xgmodel <- train(learner = lrn_tune,task = traintask)

xgpred <- predict(xgmodel,testtask)

MLmetrics::MultiLogLoss(y_true = xgpred$data$truth, y_pred = xgpred$data[,3:12])

## train the model with the previously computed parameters on the entire training set
traintask_all <- makeClassifTask(data = train_data,target = "last_event")
xgmodel_all <- train(learner = lrn_tune,task = traintask_all)

##### Scoring the model on the test data

# create the test dataset

test_all = read.csv("./data/dtest.csv", header = TRUE)[-1]
for (i in 1:ncol(test_all)){
  test_all[,i] = factor(test_all[,i])
}
test_all = cbind(test_all, dummy = 0.0)


test_all.pred = predict(xgmodel_all, newdata = test_all, type = "prob")
# testtask_all <- makeClassifTask(data = test_all,target = "last_event")


# construct output file
str(test_all.pred)
predictions = as.data.table(test_all.pred$data[1:10])
setnames(predictions, names(predictions), as.character(unique_events))
predictions[, id:=data_test_ohe$id]
str(predictions)
pred_columns <- colnames(predictions)
pred_columns <- c('id', sort(pred_columns[-length(pred_columns)]))
setcolorder(predictions, pred_columns)
setnames(predictions, c(pred_columns[1], 
                        paste('event_', pred_columns[-1], sep='')))


write.csv(predictions, file="./data/XGBoost_mlr_all_features.csv", quote=TRUE, row.names=FALSE)


parallelStop()

