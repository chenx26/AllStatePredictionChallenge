rm(list=ls())
library(Matrix)
library(data.table)		
library(xgboost)      # Extreme Gradient Boosting
library(mlr)

data_all = fread("./data/train.csv")
ids_test = fread("./data/test.csv")
# data_all = data_all[1:10000]

data_all[, dummy_counter:=1]
data_train <- subset(data_all, !(id %in% ids_test$id))
dim(data_all)
dim(data_train)
data_train[, max_timestamp_by_policy:=max(timestamp), by=id]
data_train[, min_timestamp_by_policy:=min(timestamp), by=id]
data_train_last_element = subset(data_train, timestamp == max_timestamp_by_policy)
data_train_last_element$timestamp = NULL
data_train_last_element$dummy_counter = NULL
data_train_last_element$max_timestamp_by_policy = NULL
data_train_last_element$min_timestamp_by_policy = NULL
setnames(data_train_last_element, "event", "last_event")
data_train_first_element = subset(data_train, timestamp==min_timestamp_by_policy)
data_train_first_element$timestamp = NULL
data_train_first_element$dummy_counter = NULL
data_train_first_element$max_timestamp_by_policy = NULL
data_train_first_element$min_timestamp_by_policy = NULL
setnames(data_train_first_element, "event", "first_event")

# featurize data, without the last element, by looking at counts of events
data_train_allButLast <- subset(data_train, timestamp != max_timestamp_by_policy)
data_train_ohe <- dcast(data_train_allButLast, 
                        id ~ event, 
                        fun.aggregate=sum, 
                        value.var="dummy_counter")

# create a feature that counts how many events occured in total before the last event
data_train_ohe[, total_events:=apply(data_train_ohe[, -c("id"), with=FALSE], 1, sum)]

data_train_ohe <- merge(data_train_ohe, data_train_first_element, by=c("id"))
unique_events = sort(unique(data_all$event))
data_train_ohe$first_event <- factor(data_train_ohe$first_event, levels = unique_events)
data_train_merged <- merge(data_train_ohe, data_train_last_element, by=c("id"))
# str(data_train_merged)
data_train_merged = data_train_merged[,-c("id"), with = FALSE]
colnames(data_train_merged)[1:length(unique_events)] = paste0("Event", 1:length(unique_events))
#colnames(data_train_merged)
data_train_merged$last_event = as.numeric(factor(data_train_merged$last_event, levels = unique_events))-1

train_data = sparse.model.matrix(last_event ~ . - 1, data = data_train_merged)


# prepare data for training the model

train_idx = sample(1:nrow(train_data), nrow(train_data) * 0.75)
dtrain = train_data[train_idx,]
dtest = train_data[-train_idx,]
dtrain = xgb.DMatrix(data = dtrain, label = data_train_merged$last_event[train_idx])
dtest = xgb.DMatrix(data = dtest, label = data_train_merged$last_event[-train_idx])

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

xgbcv <- xgb.cv(params = params
                ,data = dtrain
                ,nrounds = 200
                ,nfold = 5
                ,showsd = T
                ,stratified = T
                ,print.every.n = 10
                ,early.stop.round = 20
                ,maximize = F
)

xgbcv$best_iteration

#first default - model training
xgb1 <- xgb.train(
  params = params
  ,data = dtrain
  ,nrounds = xgbcv$best_iteration
  ,watchlist = list(val=dtest,train=dtrain)
  ,print.every.n = 10
  ,early.stop.round = 10
  ,maximize = F
)

#model prediction
numberOfClasses = length(unique_events)
test_pred <- predict(xgb1, dtest)
# MLmetrics::MultiLogLoss(y_true = test_pred$data$truth, y_pred = test_pred$data[,3:12])

### parameter tuning
train = data_train_merged[train_idx]
train$first_event = as.character(train$first_event)
train$last_event = as.character(train$last_event)
test = data_train_merged[-train_idx]
test$first_event = as.character(test$first_event)
test$last_event = as.character(test$last_event)
fact_col <- colnames(train)[sapply(train,is.character)]
for(i in fact_col)
  set(train,j=i,value = factor(train[[i]]))
for(i in fact_col)
  set(test,j=i,value = factor(test[[i]]))
traintask <- makeClassifTask(data = train,target = "last_event")
testtask <- makeClassifTask(data = test,target = "last_event")
lrn <- makeLearner("classif.xgboost",predict.type = "prob")
lrn$par.vals <- list(
  objective="multi:softprob",
  eval_metric="mlogloss",
  nrounds=xgbcv$best_iteration,
  eta=0.1,
  num_class = length(unique_events)
)

params <- makeParamSet(
  makeDiscreteParam("booster",values = c("gbtree","gblinear")),
  makeIntegerParam("max_depth",lower = 3L,upper = 10L),
  makeNumericParam("min_child_weight",lower = 1L,upper = 10L),
  makeNumericParam("subsample",lower = 0.5,upper = 1),
  makeNumericParam("colsample_bytree",lower = 0.5,upper = 1)
)

rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)
ctrl <- makeTuneControlRandom(maxit = 10L)

#set parallel backend
library(parallel)
library(parallelMap)
parallelStartSocket(cpus = 3)

#parameter tuning
mytune <- tuneParams(learner = lrn
                     ,task = traintask
                     ,resampling = rdesc
                     ,measures = logloss
                     ,par.set = params
                     ,control = ctrl
                     ,show.info = T)

mytune$y

lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)

xgmodel <- train(learner = lrn_tune,task = traintask)

xgpred <- predict(xgmodel,testtask)
MLmetrics::MultiLogLoss(y_true = xgpred$data$truth, y_pred = xgpred$data[,3:12])

## train the model with the previously computed parameters on the entire training set
train_all = data_train_merged
train_all$first_event = as.character(train_all$first_event)
train_all$last_event = as.character(train_all$last_event)
fact_col <- colnames(train_all)[sapply(train_all,is.character)]
for(i in fact_col)
  set(train_all,j=i,value = factor(train_all[[i]]))
traintask_all <- makeClassifTask(data = train_all,target = "last_event")
xgmodel_all <- train(learner = lrn_tune,task = traintask_all)

##### Scoring the model on the test data

# create the test dataset
data_test <- subset(data_all, id %in% ids_test$id)
data_test[, min_timestamp_by_policy:=min(timestamp), by=id]

# create the first event feature
data_test_first_event <- subset(data_test, timestamp==min_timestamp_by_policy)
data_test_first_event$timestamp = NULL
data_test_first_event$dummy_counter = NULL
data_test_first_event$min_timestamp_by_policy = NULL
setnames(data_test_first_event, "event", "first_event")

data_test_ohe <- dcast(data_test, id~event, fun.aggregate=sum, value.var="dummy_counter")
# add the total events feature
data_test_ohe[, total_events:=apply(data_test_ohe[, -c("id"), with=FALSE], 1, sum)]

# add the first event feature
data_test_ohe <- merge(data_test_ohe, data_test_first_event, by=c("id"))

unique_events = sort(unique(data_all$event))
test_ids = data_test_ohe$id
test_all = data_test_ohe[, -c("id"), with = FALSE]
test_all$first_event = as.character(test_all$first_event)
fact_col <- colnames(test_all)[sapply(test_all,is.character)]
for(i in fact_col)
  set(test_all,j=i,value = factor(test_all[[i]]))

test_all.pred = predict(xgmodel_all, newdata = test_all, type = "prob")
# testtask_all <- makeClassifTask(data = test_all,target = "last_event")
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


write.csv(predictions, file="./data/XGBoost_mlr.csv", quote=TRUE, row.names=FALSE)


parallelStop()

