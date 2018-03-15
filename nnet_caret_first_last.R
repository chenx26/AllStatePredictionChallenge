rm(list=ls())
library(Matrix)
library(data.table)		
library(xgboost)      # Extreme Gradient Boosting
# library(mlr)
library(caret)
library(doMC)
registerDoMC(cores = 8)

data_all = fread("./data/train.csv")
ids_test = fread("./data/test.csv")
# data_all = data_all[1:100000]

data_all[, dummy_counter:=1]
data_train <- subset(data_all, !(id %in% ids_test$id))
dim(data_all)
dim(data_train)
data_train[, max_timestamp_by_policy:=max(timestamp), by=id]
data_train[, second_max_timestamp_by_policy:=sort(timestamp,partial=length(timestamp)-1)[length(timestamp)-1], by=id]
data_train[, min_timestamp_by_policy:=min(timestamp), by=id]
data_train_last_element = subset(data_train, timestamp == max_timestamp_by_policy)
data_train_last_element$timestamp = NULL
data_train_last_element$dummy_counter = NULL
data_train_last_element$max_timestamp_by_policy = NULL
data_train_last_element$second_max_timestamp_by_policy = NULL
data_train_last_element$min_timestamp_by_policy = NULL
setnames(data_train_last_element, "event", "last_event")
data_train_first_element = subset(data_train, timestamp==min_timestamp_by_policy)
data_train_first_element$timestamp = NULL
data_train_first_element$dummy_counter = NULL
data_train_first_element$max_timestamp_by_policy = NULL
data_train_first_element$second_max_timestamp_by_policy = NULL
data_train_first_element$min_timestamp_by_policy = NULL
setnames(data_train_first_element, "event", "first_event")
data_train_second_last_element = subset(data_train, timestamp == second_max_timestamp_by_policy)
data_train_second_last_element$timestamp = NULL
data_train_second_last_element$dummy_counter = NULL
data_train_second_last_element$max_timestamp_by_policy = NULL
data_train_second_last_element$second_max_timestamp_by_policy = NULL
data_train_second_last_element$min_timestamp_by_policy = NULL
setnames(data_train_second_last_element, "event", "second_last_event")

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
unique_events = sort(unique(data_all$event))
data_train_ohe$first_event <- factor(data_train_ohe$first_event)
data_train_ohe$second_last_event <- factor(data_train_ohe$second_last_event)
data_train_merged <- merge(data_train_ohe, data_train_last_element, by=c("id"))
# str(data_train_merged)
data_train_merged = data_train_merged[,-c("id"), with = FALSE]
colnames(data_train_merged)[1:length(unique_events)] = paste0("Event", 1:length(unique_events))
#colnames(data_train_merged)
data_train_merged$last_event = factor(data_train_merged$last_event)

### parameter tuning
train_idx = sample(1:nrow(data_train_merged), nrow(data_train_merged)*0.75)
dtrain = data_train_merged[train_idx]
dtrain$first_event = as.character(dtrain$first_event)
dtrain$second_last_event = as.character(dtrain$second_last_event)
dtrain$last_event = as.character(dtrain$last_event)
fact_col <- colnames(dtrain)[sapply(dtrain,is.character)]
for(i in fact_col)
  dtrain[[i]] = factor(sapply(dtrain[[i]], function(x) paste("E_", x, sep ="")))
dtest = data_train_merged[-train_idx]
dtest$first_event = as.character(dtest$first_event)
dtest$second_last_event = as.character(dtest$second_last_event)
dtest$last_event = as.character(dtest$last_event)
fact_col <- colnames(dtest)[sapply(dtest,is.character)]
for(i in fact_col)
  dtest[[i]] = factor(sapply(dtest[[i]], function(x) paste("E_", x, sep ="")))

# model <- train(last_event~., data=dtrain, method='nnet', trControl=trainControl(method='cv'))


numFolds <- trainControl(method = 'cv', number = 5, classProbs = TRUE, verboseIter = TRUE, summaryFunction = mnLogLoss, preProcOptions = list(thresh = 0.75, ICAcomp = 3, k = 5))
fit2 <- caret::train(last_event ~ ., data = dtrain, method = 'nnet', trControl = numFolds, tuneGrid=expand.grid(size=c(20), decay=c(0.5,0.6,0.7,0.8,0.9)))

probs <- predict(fit2, newdata=dtest[,-c("last_event"), with = FALSE], type='prob')
MLmetrics::MultiLogLoss(y_true = dtest$last_event, y_pred = probs)





##### Scoring the model on the test data

# create the test dataset
data_test <- subset(data_all, id %in% ids_test$id)
data_test[, min_timestamp_by_policy:=min(timestamp), by=id]
data_test[, second_max_timestamp_by_policy:=max(timestamp), by=id]

data_test_first_event = subset(data_test, timestamp==min_timestamp_by_policy)
data_test_first_event$timestamp = NULL
data_test_first_event$dummy_counter = NULL
data_test_first_event$second_max_timestamp_by_policy = NULL
data_test_first_event$min_timestamp_by_policy = NULL
setnames(data_test_first_event, "event", "first_event")

data_test_second_last_event = subset(data_test, timestamp == second_max_timestamp_by_policy)
data_test_second_last_event$timestamp = NULL
data_test_second_last_event$dummy_counter = NULL
data_test_second_last_event$second_max_timestamp_by_policy = NULL
data_test_second_last_event$min_timestamp_by_policy = NULL
setnames(data_test_second_last_event, "event", "second_last_event")


data_test_ohe <- dcast(data_test, id~event, fun.aggregate=sum, value.var="dummy_counter")
# add the total events feature
data_test_ohe[, total_events:=apply(data_test_ohe[, -c("id"), with=FALSE], 1, sum)]

# add the first event feature
data_test_ohe <- merge(data_test_ohe, data_test_first_event, by=c("id"))
data_test_ohe <- merge(data_test_ohe, data_test_second_last_event, by=c("id"))

unique_events = sort(unique(data_all$event))
test_ids = data_test_ohe$id
test_all = data_test_ohe[, -c("id"), with = FALSE]
test_all$first_event = as.character(test_all$first_event)
test_all$second_last_event = as.character(test_all$second_last_event)
fact_col <- colnames(test_all)[sapply(test_all,is.character)]
for(i in fact_col)
  test_all[[i]] = factor(sapply(test_all[[i]], function(x) paste("E_", x, sep ="")))
colnames(test_all)[1:length(unique_events)] = paste0("Event", 1:length(unique_events))

test_all.pred = predict(fit2, newdata = test_all, type = "prob")
# testtask_all <- makeClassifTask(data = test_all,target = "last_event")
str(test_all.pred)
predictions = as.data.table(test_all.pred[1:10])
setnames(predictions, names(predictions), as.character(unique_events))
predictions[, id:=data_test_ohe$id]
str(predictions)
pred_columns <- colnames(predictions)
pred_columns <- c('id', sort(pred_columns[-length(pred_columns)]))
setcolorder(predictions, pred_columns)
setnames(predictions, c(pred_columns[1], 
                        paste('event_', pred_columns[-1], sep='')))


write.csv(predictions, file="./data/nnet_caret_first_last_size_20_tune_decay.csv", quote=TRUE, row.names=FALSE)

