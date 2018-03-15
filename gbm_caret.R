rm(list=ls())
library(data.table)
library(doParallel)
# library(gbm)
library(caret)

data_in = fread("./data/train.csv")
ids_test = fread("./data/test.csv")

data_all = data_in[1:1000]

data_all
ids_test

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

data_train_ohe

data_train_ohe <- merge(data_train_ohe, data_train_first_element, by=c("id"))
data_train_ohe$first_event <- as.factor(data_train_ohe$first_event)

data_train_merged <- merge(data_train_ohe, data_train_last_element, by=c("id"))

data_train_merged

# find the optimal gbm model with caret

fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10)

gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = (1:30)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

nrow(gbmGrid)

registerDoParallel(4)		# Registrer a parallel backend for train
getDoParWorkers()

model <- train(last_event ~ . - id , data = data_train_merged, 
                 method = "gbm", 
                 trControl = fitControl, 
                 verbose = FALSE, 
                 ## Now specify the exact models 
                 ## to evaluate:
                 tuneGrid = gbmGrid, distribution = "multinomial")

# train a gbm model
# WARNING: Run time can exceed 10-15 minutes. For faster run times, lower n.trees=10
# model <- gbm(last_event ~ . - id, 
#              distribution="multinomial", 
#              data=data_train_merged, 
#              interaction.depth=5,
#              n.trees=500,
#              shrinkage=0.1,
#              train.fraction=0.8,
#              verbose=TRUE)
# 
# gbm.perf(model)
# summary(model, las=2)



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

# score the model
predictions_raw <- predict(model, data_test_ohe, type="response")

#package it into a format that can be scored
predictions <- as.data.table(predictions_raw[,,1])
setnames(predictions, names(predictions), colnames(predictions_raw))
predictions[, id:=data_test_ohe$id]

predictions

pred_columns <- colnames(predictions)
pred_columns <- c('id', sort(pred_columns[-length(pred_columns)]))
setcolorder(predictions, pred_columns)
setnames(predictions, c(pred_columns[1], 
                        paste('event_', pred_columns[-1], sep='')))

predictions

write.csv(predictions, file="./data/gbm_caret.csv", quote=TRUE, row.names=FALSE)
