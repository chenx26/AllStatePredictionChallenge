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
data_train[, second_max_timestamp_by_policy:=sort(timestamp,partial=length(timestamp)-1)[length(timestamp)-1], by=id]
data_train[, third_max_timestamp_by_policy:=sort(timestamp,partial=2)[2], by=id]
data_train[, min_timestamp_by_policy:=min(timestamp), by=id]
data_train_last_element = subset(data_train, timestamp == max_timestamp_by_policy)
data_train_last_element$timestamp = NULL
data_train_last_element$dummy_counter = NULL
data_train_last_element$max_timestamp_by_policy = NULL
data_train_last_element$second_max_timestamp_by_policy = NULL
data_train_last_element$third_max_timestamp_by_policy = NULL
data_train_last_element$min_timestamp_by_policy = NULL
setnames(data_train_last_element, "event", "last_event")
data_train_first_element = subset(data_train, timestamp==min_timestamp_by_policy)
data_train_first_element$timestamp = NULL
data_train_first_element$dummy_counter = NULL
data_train_first_element$max_timestamp_by_policy = NULL
data_train_first_element$second_max_timestamp_by_policy = NULL
data_train_first_element$third_max_timestamp_by_policy = NULL
data_train_first_element$min_timestamp_by_policy = NULL
setnames(data_train_first_element, "event", "first_event")
data_train_second_last_element = subset(data_train, timestamp == second_max_timestamp_by_policy)
data_train_second_last_element$timestamp = NULL
data_train_second_last_element$dummy_counter = NULL
data_train_second_last_element$max_timestamp_by_policy = NULL
data_train_second_last_element$second_max_timestamp_by_policy = NULL
data_train_second_last_element$third_max_timestamp_by_policy = NULL
data_train_second_last_element$min_timestamp_by_policy = NULL
setnames(data_train_second_last_element, "event", "second_last_event")

data_train_third_last_element = subset(data_train, timestamp == third_max_timestamp_by_policy)
data_train_third_last_element$timestamp = NULL
data_train_third_last_element$dummy_counter = NULL
data_train_third_last_element$max_timestamp_by_policy = NULL
data_train_third_last_element$second_max_timestamp_by_policy = NULL
data_train_third_last_element$third_max_timestamp_by_policy = NULL
data_train_third_last_element$min_timestamp_by_policy = NULL
setnames(data_train_third_last_element, "event", "third_last_event")

# featurize data, without the last element, by looking at counts of events
data_train_allButLast <- subset(data_train, timestamp != max_timestamp_by_policy)
data_train_ohe <- dcast(data_train_allButLast, 
                        id ~ event, 
                        fun.aggregate=sum, 
                        value.var="dummy_counter")

# create a feature that counts how many events occured in total before the last event
data_train_ohe[, total_events:=apply(data_train_ohe[, -c("id"), with=FALSE], 1, sum)]

data_train_ohe <- merge(data_train_ohe, data_train_first_element, by=c("id"))
data_train_ohe <- merge(data_train_ohe, data_train_second_last_element, by=c("id"))
data_train_ohe <- merge(data_train_ohe, data_train_third_last_element, by=c("id"))
unique_events = sort(unique(data_all$event))
data_train_ohe$first_event <- factor(data_train_ohe$first_event, levels = unique_events)
data_train_ohe$second_last_event <- factor(data_train_ohe$second_last_event, levels = unique_events)
data_train_ohe$third_last_event <- factor(data_train_ohe$third_last_event, levels = unique_events)
data_train_merged <- merge(data_train_ohe, data_train_last_element, by=c("id"))
# str(data_train_merged)
data_train_merged = data_train_merged[,-c("id"), with = FALSE]
colnames(data_train_merged)[1:length(unique_events)] = paste0("Event", 1:length(unique_events))
#colnames(data_train_merged)
data_train_merged$last_event = as.numeric(factor(data_train_merged$last_event, levels = unique_events))-1

train_data = sparse.model.matrix(last_event ~ . - 1, data = data_train_merged)
str(train_data)
colnames(train_data)
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

#set parallel backend
library(doMC)
registerDoMC(cores = 4)

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
MLmetrics::MultiLogLoss(y_true = data_train_merged$last_event[-train_idx], y_pred = matrix(test_pred, nrow = nrow(dtest), byrow = TRUE))

### train model on all training data

xgb_all <- xgb.train(
  params = params
  ,data = xgb.DMatrix(data = train_data, label = data_train_merged$last_event)
  ,nrounds = xgbcv$best_iteration
  ,watchlist = list(val=dtest,train=dtrain)
  ,print.every.n = 10
#  ,early.stop.round = 10
  ,maximize = F
)


##### Scoring the model on the test data

# create the test dataset
data_test <- subset(data_all, id %in% ids_test$id)
data_test[, min_timestamp_by_policy:=min(timestamp), by=id]
data_test[, second_max_timestamp_by_policy:=max(timestamp), by=id]
data_test[, third_max_timestamp_by_policy:=sort(timestamp,partial=2)[2], by=id]

data_test_first_event = subset(data_test, timestamp==min_timestamp_by_policy)
data_test_first_event$timestamp = NULL
data_test_first_event$dummy_counter = NULL
data_test_first_event$second_max_timestamp_by_policy = NULL
data_test_first_event$third_max_timestamp_by_policy = NULL
data_test_first_event$min_timestamp_by_policy = NULL
setnames(data_test_first_event, "event", "first_event")

data_test_second_last_event = subset(data_test, timestamp == second_max_timestamp_by_policy)
data_test_second_last_event$timestamp = NULL
data_test_second_last_event$dummy_counter = NULL
data_test_second_last_event$second_max_timestamp_by_policy = NULL
data_test_second_last_event$third_max_timestamp_by_policy = NULL
data_test_second_last_event$min_timestamp_by_policy = NULL
setnames(data_test_second_last_event, "event", "second_last_event")

data_test_third_last_event = subset(data_test, timestamp == third_max_timestamp_by_policy)
data_test_third_last_event$timestamp = NULL
data_test_third_last_event$dummy_counter = NULL
data_test_third_last_event$second_max_timestamp_by_policy = NULL
data_test_third_last_event$third_max_timestamp_by_policy = NULL
data_test_third_last_event$min_timestamp_by_policy = NULL
setnames(data_test_third_last_event, "event", "third_last_event")


data_test_ohe <- dcast(data_test, id~event, fun.aggregate=sum, value.var="dummy_counter")
# add the total events feature
data_test_ohe[, total_events:=apply(data_test_ohe[, -c("id"), with=FALSE], 1, sum)]

# add the first event feature
data_test_ohe <- merge(data_test_ohe, data_test_first_event, by=c("id"))
data_test_ohe <- merge(data_test_ohe, data_test_second_last_event, by=c("id"))
data_test_ohe <- merge(data_test_ohe, data_test_third_last_event, by=c("id"))

test_ids = data_test_ohe$id
data_test_ohe$first_event <- factor(data_test_ohe$first_event, levels = unique_events)
data_test_ohe$second_last_event <- factor(data_test_ohe$second_last_event, levels = unique_events)
data_test_ohe$third_last_event <- factor(data_test_ohe$third_last_event, levels = unique_events)
colnames(data_test_ohe)[1:length(unique_events)+1] = paste0("Event", 1:length(unique_events))
# test_all = data_test_ohe[, -c("id"), with = FALSE]
dtest_all = sparse.model.matrix(id ~ . - 1, data = data_test_ohe)


test_all.pred = predict(xgb_all, dtest_all)
test_all.pred.mat = matrix(test_all.pred, nrow = nrow(dtest_all), byrow = TRUE)
# testtask_all <- makeClassifTask(data = test_all,target = "last_event")
str(test_all.pred)
predictions = as.data.table(test_all.pred.mat)
setnames(predictions, names(predictions), as.character(unique_events))
predictions[, id:=data_test_ohe$id]
str(predictions)
pred_columns <- colnames(predictions)
pred_columns <- c('id', sort(pred_columns[-length(pred_columns)]))
setcolorder(predictions, pred_columns)
setnames(predictions, c(pred_columns[1], 
                        paste('event_', pred_columns[-1], sep='')))


write.csv(predictions, file="./data/XGBoost_no_mlr_first_first2_last.csv", quote=TRUE, row.names=FALSE)


parallelStop()

