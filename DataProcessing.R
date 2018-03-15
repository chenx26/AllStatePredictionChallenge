rm(list=ls())
library(data.table)
# This is the code to build the training matrix, validation vector and the test matrix

# read all data
# for the train set, the number of events go from 2 to 25, excluding the last event
# for the test set, the number of events go from 2 to 23

train_number_of_features = 25
test_number_of_features = 23
empty_code = 11111

data_all = fread("./data/train.csv")
# data_all = data_all[1:10000]
# read test ids
ids_test = fread("./data/test.csv")

# data_all = data_all[1:10000]

data_all[, dummy_counter:=1]
data_train <- subset(data_all, !(id %in% ids_test$id))
data_train[, max_timestamp_by_policy:=max(timestamp), by=id]

# find the last event and save in separate data.table
data_train_last_element = subset(data_train, timestamp == max_timestamp_by_policy)
data_train_last_element$timestamp = NULL
data_train_last_element$max_timestamp_by_policy = NULL
setnames(data_train_last_element, "event", "last_event")

# save last_event
write.csv(data_train_last_element$last_event, "./data/last_event.csv")

# remove the last event from the data_train
data_train_allButLast <- subset(data_train, timestamp != max_timestamp_by_policy)

dtrain = matrix(empty_code, nrow = length(unique(data_train$id)), ncol = train_number_of_features)
colnames(dtrain) = paste0("Event", 1:train_number_of_features)
rownames(dtrain) = unique(data_train$id)
for(i in 1:nrow(data_train_allButLast)){
  id = data_train_allButLast$id[i]
  idx = min(which(dtrain[id,]==empty_code))
  dtrain[id,idx] = data_train_allButLast$event[i]
  if (i%%1000==0){
    print(i)
  }
}

write.csv(dtrain, "./data/dtrain.csv")


data_test = subset(data_all, (id %in% ids_test$id))
dtest = matrix(empty_code, nrow = length(unique(data_test$id)), ncol = train_number_of_features)
colnames(dtest) = paste0("Event", 1:train_number_of_features)
rownames(dtest) = unique(data_test$id)
for(i in 1:nrow(data_test)){
  id = data_test$id[i]
  idx = min(which(dtest[id,]==empty_code))
  dtest[id,idx] = data_test$event[i]
}

write.csv(dtest, "./data/dtest.csv")
        