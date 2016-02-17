setwd("/Users/knguyen1/Documents/InnocentiveDataChallenge/Innocentive-Data-Challenge-Project")
library(readr)
library(xgboost)
library(rpart)
library(rpart.plot)
library(e1071)
library(randomForest)
library(RRF)
library(gbm)
library(foreach)
library(quint)

#Set a random seed for reproducibility 
set.seed(1)

#Reading in the training and testing data
train = read.csv("Innocentive_Training_Data.csv")
test = read.csv("Innocentive_Testing_Data.csv")
submission = read.csv('Innocentive_Example_Submission.csv')

#Setup the classification for the training datasets:
train$Outcome = 0
train$Outcome[(train$dataset==1)] = 0
train$Outcome[(train$dataset==2)&(train$x29 < 51.3)] = 1
train$Outcome[(train$dataset==3)&(train$x5 == 1 | train$x5 == 2)] = 1
train$Outcome[(train$dataset==4)&(train$x4 == 1 | train$x4 == 2)&(train$x22 > 57.7)] = 1


#####################################
#Data cleaning
#Normalize the values in all columns
#for (i in colnames(train)[5:24]){
#  train[[i]] = as.factor(train[[i]])
#}

#for (i in colnames(test)[5:24]){
#  test[[i]] = as.factor(test[[i]])
#}

for (i in colnames(train)[25:44]){
  train[[i]] = train[[i]]/100
}

for (i in colnames(test)[25:44]){
  test[[i]] = test[[i]]/100
}

#train$x5 = as.factor(train$x5)
#test$x5 = as.factor(test$x5)

#for (i in colnames(train)[25:44]){
#  for (j in colnames(train)[25:44]){
#    train[[paste(i, j, sep="")]] = train[[i]]*train[[j]]
#  }
#}

#for (i in colnames(test)[25:44]){
#  for (j in colnames(test)[25:44]){
#    test[[paste(i, j, sep="")]] = test[[i]]*test[[j]]
#  }
#}

#for (i in colnames(train)[25:44]){
#  train[[i]] = log(train[[i]])
#}

#for (i in colnames(test)[25:44]){
#  test[[i]] = log(test[[i]])
#}

#Normalize the values of y column
#train$y = train$y/max(train$y)
#test$y = test$y/max(test$y)

#Minmax scaling for the values of the y column
train$y = train$y/max(train$y)
test$y = test$y/max(test$y)

#Log scaling for the values of the y column
#train$y = log(train$y)
#test$y = log(test$y)

#Set the Outcome as factor variable
#train$Outcome = as.factor(train$Outcome)

#############################################################
#Trying some feature engineering
train$index2229 = (train$x22*train$x29)
test$index2229 = (test$x22*test$x29)

#train$squarex22 = train$x22^2
#test$squarex22 = test$x22^2

#train$squarex29 = train$x29^2
#test$squarex29 = test$x29^2

#train$index3536 = (train$x35*train$x36)
#test$index3536 = (test$x35*test$x36)

#Redundant features: x3, x6, x8, x9, x11, x16, x17, x30, x34, x35, x37, x39

#removedFeatures = x11 + x19 + x10 + x13 + x15 + x23 + x24 + x25 + x26 + x30 + x31 + x34 + x37 + x38 + x39 + x40

#Split the training data sets into two
first_train = train[seq(1,959,2),]
second_train = train[seq(2,960,2),]

#train$Outcome = as.factor(train$Outcome)

#############################################################
#RandomForest classifier
#----------------------------Two ensemble models take a vote---------------------------- 
RandomForest1 = randomForest(Outcome ~ . -x11, data=train, ntree=7, type='class', importance=FALSE)

#Boost1 = gbm(Outcome ~., data=train, distribution="bernoulli", n.trees = 100, interaction.depth = 6)

RandomForest2 = randomForest(Outcome ~ . -x11, data=train, ntree=8, type='class', importance=FALSE)

#RandomForest3 = randomForest(Outcome ~. -x11, data=train, ntree=6, type='class', importance=FALSE)
#RandomForest4 = randomForest(Outcome ~., data=train, ntree=6, type='class', importance=FALSE)
#RandomForest5 = randomForest(Outcome ~., data=train, ntree=7, type='class', importance=FALSE)
#RandomForest6 = randomForest(Outcome ~., data=train, ntree=8, type='class', importance=FALSE)
#RandomForest7 = randomForest(Outcome ~., data=train, ntree=9, type='class', importance=FALSE)
#RandomForest8 = randomForest(Outcome ~., data=train, ntree=10, type='class', importance=FALSE)
#CombinedForest = combine(RandomForest1, RandomForest2)
#RandomForest9 = randomForest(Outcome ~., data=train, ntree=11, type='class', importance=FALSE)
#RandomForest10 = randomForest(Outcome ~., data=train, ntree=12, type='class', importance=FALSE)

#----------------------------------------QUINT classifier-------------
#Quint1 = quint(data=train)
cat("making predictions\n")

#Support Vector classifier
#length_divisor<-4  

#iterations<-50

#predictions <- foreach(m=1:iterations,.combine=cbind) %do% {  
#  training_positions <- sample(nrow(train), size=floor((nrow(train)/length_divisor)))  
#  train_pos<-1:nrow(train) %in% training_positions  
#  svm_fit<-svm(Outcome~.,data=train[train_pos,])  
#  predict(svm_fit,newdata=test)  
#}  

#Form a preliminary submission
submission <- data.frame(Id=c(1:288000))
prediction1 <- as.integer(predict(RandomForest1, newdata=test))
prediction2 <- as.integer(predict(RandomForest2, newdata=test))
prediction = ceiling((prediction1 + prediction2)/2)
#prediction = ceiling((as.integer(prediction1) + as.integer(prediction2) - 2)/2)
#test$Outcome <- as.factor(prediction)
submission$Outcome = prediction

#---------------------Ensemble model (30 models)----------------------------- 
#length_divisor<-1.5  

#iterations<-20

#prediction1 <- foreach(m=1:iterations,.combine=cbind) %do% {  
  #training_positions <- sample(nrow(train), size=floor((nrow(train)/length_divisor)))  
  #train_pos<-1:nrow(train) %in% training_positions  
  #RandomForest1 = randomForest(Outcome ~ . -x11, data=train, ntree=1+2*m, type='class', importance=FALSE, mtry=14) 
  #predict(RandomForest1,newdata=test)  
#}  

#prediction1 <- prediction1 - 1

#RF_prediction1 = apply(prediction1, 1, mean)

#prediction2 <- foreach(m=1:iterations,.combine=cbind) %do% {  
  #training_positions <- sample(nrow(train), size=floor((nrow(train)/length_divisor)))  
  #train_pos<-1:nrow(train) %in% training_positions  
  #RandomForest2 = randomForest(Outcome ~ . -x11, data=train, ntree=2+2*m, type='class', importance=FALSE, mtry=15) 
  #predict(RandomForest2,newdata=test)  
#}

#prediction2 <- prediction2 - 1
#RF_prediction2 = apply(prediction2, 1, mean)


#prediction = ceiling((RF_prediction1 + RF_prediction2)/2 - 0.5)
#test$Outcome <- as.factor(prediction)
#submission$Outcome = test$Outcome



##################################################################
#Obtain final submission
finalsubmission = data.frame(id=c(1:240))
for(i in unique(ceiling(submission$Id/240))){
  finalsubmission[toString(i)] = submission$Outcome[ceiling(submission$Id/240) == i]
}

names(finalsubmission) = gsub("^", 'dataset_', names(finalsubmission))
names(finalsubmission) = gsub("dataset_id", 'id', names(finalsubmission))

finalsubmission$dataset_2 = train$Outcome[241:480]
finalsubmission$dataset_3 = train$Outcome[481:720]
finalsubmission$dataset_4 = train$Outcome[721:960]

cat("saving the submission file\n")
write_csv(finalsubmission, "randomforest_submission.csv")

print(length(which(train$Outcome[721:960] == finalsubmission$dataset_4)))
print(length(which(train$Outcome[481:720] == finalsubmission$dataset_3)))
print(length(which(train$Outcome[241:480] == finalsubmission$dataset_2)))
print(length(which(train$Outcome[1:240] == finalsubmission$dataset_1)))





