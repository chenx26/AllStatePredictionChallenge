rm(list=ls())
library(Matrix)
library(data.table)
library(caret)
library(corrplot)			# plot correlations
library(doParallel)		# parallel processing
library(dplyr)        # Used by caret
library(gbm)				  # GBM Models
library(pROC)				  # plot the ROC curve
library(xgboost)      # Extreme Gradient Boosting



data_all = fread("./data/train.csv")
ids_test = fread("./data/test.csv")
data_all = data_all[1:10000]

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
str(data_train_merged)
data_train_merged = data_train_merged[,-c("id"), with = FALSE]
colnames(data_train_merged)[1:length(unique_events)] = paste0("Event", 1:length(unique_events))
colnames(data_train_merged)
sparse_matrix = sparse.model.matrix(last_event ~ . - 1, data = data_train_merged)
data_train_merged$last_event = as.numeric(factor(data_train_merged$last_event, levels = unique_events))-1

##----------------------------------------------
## XGBOOST
# Some stackexchange guidance for xgboost
# http://stats.stackexchange.com/questions/171043/how-to-tune-hyperparameters-of-xgboost-trees

# Set up for parallel procerssing
ctrl <- trainControl(method = "repeatedcv",   # 10fold cross validation
                     number = 3,							# do 5 repititions of cv
                     summaryFunction=mnLogLoss,	# Use AUC to pick the best model
                     classProbs=TRUE,
                     allowParallel = FALSE)

set.seed(1951)
registerDoParallel(3,cores=3)
getDoParWorkers()

# Train xgboost
xgb.grid <- expand.grid(nrounds = 100, #the maximum number of iterations
                        eta = c(0.01,0.1), # shrinkage
                        max_depth = c(2,6,10),
                        gamma = c(0,0.1),
                        colsample_bytree = 1,
                        min_child_weight = 1,
                        subsample = 1)

xgb.tune <-train(x=sparse_matrix,y=data_train_merged$last_event,
                 method="xgbTree",
                 metric="mlogloss",
                 tuneGrid=xgb.grid,
                 num_class = length(unique_events), 
                 objective = "multi:softprob", nthread = 3)


xgb.tune$bestTune
plot(xgb.tune)  		# Plot the performance of the training models
res <- xgb.tune$results
res

### xgboostModel Predictions and Performance
# Make predictions using the test data set
xgb.pred <- predict(xgb.tune,testX)

#Look at the confusion matrix  
confusionMatrix(xgb.pred,testData$Class)   

#Draw the ROC curve 
xgb.probs <- predict(xgb.tune,testX,type="prob")
#head(xgb.probs)

xgb.ROC <- roc(predictor=xgb.probs$PS,
               response=testData$Class,
               levels=rev(levels(testData$Class)))
xgb.ROC$auc
# Area under the curve: 0.8857

plot(xgb.ROC,main="xgboost ROC")
# Plot the propability of poor segmentation
histogram(~xgb.probs$PS|testData$Class,xlab="Probability of Poor Segmentation")


# Comparing Multiple Models
# Having set the same seed before running gbm.tune and xgb.tune
# we have generated paired samples and are in a position to compare models 
# using a resampling technique.
# (See Hothorn at al, "The design and analysis of benchmark experiments
# -Journal of Computational and Graphical Statistics (2005) vol 14 (3) 
# pp 675-699) 

rValues <- resamples(list(xgb=xgb.tune,gbm=gbm.tune))
rValues$values
summary(rValues)

bwplot(rValues,metric="ROC",main="GBM vs xgboost")	# boxplot
dotplot(rValues,metric="ROC",main="GBM vs xgboost")	# dotplot
#splom(rValues,metric="ROC")