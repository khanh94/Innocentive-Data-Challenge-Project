setwd("/Users/knguyen1/Documents/InnocentiveDataChallenge/Innocentive-Data-Challenge-Project")
library(readr)
library(xgboost)
library(rpart)
library(rpart.plot)

#Set a random seed for reproducibility 
set.seed(1)

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

feature.names <- as.factor(names(train)[5:ncol(train)-1])

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

#DecisionTree classifier
ClinicalTree = rpart(Score ~., data=train, method="class", minbucket=25)
prp(ClinicalTree)
cat("making predictions\n")

#Form a preliminary submission
submission <- data.frame(Id=c(1:288000))
test$Score <- as.integer(predict(ClinicalTree, newdata=test, type='class'))
submission$Score = test$Score - 1


#Obtain final submission
finalsubmission = data.frame(id=c(1:240))
for(i in unique(ceiling(submission$Id/240))){
  finalsubmission[toString(i)] = submission$Score[ceiling(submission$Id/240) == i]
}

names(finalsubmission) = gsub("^", 'dataset_', names(finalsubmission))
names(finalsubmission) = gsub("dataset_id", 'id', names(finalsubmission))


cat("saving the submission file\n")
write_csv(finalsubmission, "decisiontree_submission.csv")


#Some details about the trained tree
#tree = xgb.dump(xgb, with.stats=T)
#tree[1:10]





