setwd("/Users/knguyen1/Documents/InnocentiveDataChallenge")
library(readr)
library(xgboost)
library(rpart)
library(rpart.plot)

#Set a random seed for reproducibility 
set.seed(8)

#Reading in the training and testing data
train = read.csv("Innocentive_Training_Data.csv")
test = read.csv("Innocentive_Testing_Data.csv")
submission = read.csv('Innocentive_Example_Submission.csv')

#Setup the classification for the training datasets:
train$Score = 0
train$Score[(train$dataset==1)] = 0
train$Score[(train$dataset==2)&(train$x29 < 51.3)] = 1
train$Score[(train$dataset==3)&(train$x5 == 1 | train$x5 == 2)] = 1
train$Score[(train$dataset==4)&(train$x4 == 1 | train$x4 == 2)&(train$x22 > 57.7)] = 1

feature.names <- names(train)[5:ncol(train)-1]

#Data cleaning
#Normalize the values in all columns
for (i in colnames(train)[25:44]){
  train[[i]] = train[[i]]/max(train[[i]])
}

for (i in colnames(test)[25:44]){
  test[[i]] = test[[i]]/max(test[[i]])
}

#Normalize the values of y column
train$y = train$y/max(train$y)
test$y = test$y/max(test$y)

#Build a classification model based on training data (xgboost, 33.14)
cat("training a XGBoost classifier\n")
clf <- xgboost(data        = data.matrix(train[,feature.names]),
               label       = train$Score,
               eta         = 0.04,
               depth       = 25,
               nrounds     = 250,
               objective   = "binary:logistic",
               eval_metric = "error", 
)

#DecisionTree classifier
#ClinicalTree = rpart(Score, data=train, method="class", minbucket=25)
#prp(ClinicalTree)
#cat("making predictions\n")

#Form a preliminary submission
submission <- data.frame(Id=c(1:288000))
test$Score <- as.integer(round(predict(clf, newdata=data.matrix(test[,feature.names]))))
submission$Score = test$Score
submission$Score[1:240] = 0

#Obtain final submission
finalsubmission = data.frame(id=c(1:240))
for(i in unique(ceiling(submission$Id/240))){
    finalsubmission[toString(i)] = submission$Score[ceiling(submission$Id/240) == i]
}

names(finalsubmission) = gsub("^", 'dataset_', names(finalsubmission))
names(finalsubmission) = gsub("dataset_id", 'id', names(finalsubmission))


cat("saving the submission file\n")
write_csv(finalsubmission, "xgboost_submission.csv")


#Some details about the trained tree
#tree = xgb.dump(xgb, with.stats=T)
#tree[1:10]





