# Splitting sentiment column into 3 columns - positives, negatives and neutrals - to mimic One-vs-Rest heuristic approach
# and enable logistic regression (a binary classifier) to perform multiclass classification

################################################################################
# POSITIVES PREDICTION
################################################################################

train <- read.csv("train.csv",stringsAsFactors=FALSE)

train$Pos <- as.factor(train$sentiment==3)
train$Pos <- ifelse(train$Pos == TRUE, 1, 0)

library(tm)
train_corpus <- Corpus(VectorSource(train$tweet))

# Data pre-processing of train corpus
train_corpus <- tm_map(train_corpus, content_transformer(tolower))
train_corpus <- tm_map(train_corpus,removeWords,stopwords("english"))
train_corpus <- tm_map(train_corpus,removePunctuation)
library(SnowballC)
train_corpus <- tm_map(train_corpus,stemDocument)

train_dtm <- DocumentTermMatrix(train_corpus)
#train_dtm

train_dtm <- removeSparseTerms(train_dtm,0.995)

trainsparse <- as.data.frame(as.matrix(train_dtm))
colnames(trainsparse) <- make.names(colnames(trainsparse))
head(trainsparse)

trainsparse$Pos <- train$Pos

set.seed(123) 
library(caTools)
spl <- sample.split(trainsparse$Pos,SplitRatio=0.7) 
train <- subset(trainsparse,spl==TRUE);             
valid <- subset(trainsparse,spl==FALSE);             

model1 <- glm(Pos~.,data=train,family=binomial)
summary(model1)

predict1 <- predict(model1,newdata=train,type="response")
predict2 <- predict(model1,newdata=valid,type="response")

accuracy <- function(predict_object, data, threshold=0.5) {
  return(sum(diag(table(predict_object >= threshold, data))) /
           sum(table(predict_object >= threshold, data)))
}

accuracy(predict1, train$Pos)
accuracy(predict2, valid$Pos)

# The accuracy of logistic regression model in the train data is 0.8366258
# The accuracy of logistic regression model in the validation data is 0.8295326

# Predicting on test data
test <- read.csv("test.csv")

library(tm)
test_corpus <- Corpus(VectorSource(test$tweet))

# Data pre-processing of test corpus
test_corpus <- tm_map(test_corpus, content_transformer(tolower))
test_corpus <- tm_map(test_corpus,removeWords,stopwords("english"))
test_corpus <- tm_map(test_corpus,removePunctuation)
library(SnowballC)
test_corpus <- tm_map(test_corpus,stemDocument)

test_dtm <- DocumentTermMatrix(test_corpus)

test_dtm <- removeSparseTerms(test_dtm,0.995)

new_test <- as.data.frame(as.matrix(test_dtm))
colnames(new_test) <- make.names(colnames(new_test))

library(tidyverse)

common_colnames <- intersect(names(trainsparse), names(new_test))
new_test <- select(new_test, common_colnames)
new_train <- select(trainsparse, common_colnames)
new_train$Pos <- trainsparse$Pos

model2 <- glm(Pos~.,data=new_train,family=binomial)

Pos_predict <- predict(model2,newdata=new_test,type="response")

Pos_predict <- as.data.frame(Pos_predict)
Pos_predict$Pos_binary <- ifelse(Pos_predict$Pos_predict >= 0.5, 3, 0)
table(Pos_predict$Pos_binary)
head(Pos_predict)

################################################################################
# NEUTRALS PREDICTION
################################################################################

train <- read.csv("train.csv",stringsAsFactors=FALSE)

train$Neu <- as.factor(train$sentiment==2)
train$Neu <- ifelse(train$Neu == TRUE, 1, 0)

library(tm)
train_corpus <- Corpus(VectorSource(train$tweet))

# Data pre-processing of train corpus
train_corpus <- tm_map(train_corpus, content_transformer(tolower))
train_corpus <- tm_map(train_corpus,removeWords,stopwords("english"))
train_corpus <- tm_map(train_corpus,removePunctuation)
library(SnowballC)
train_corpus <- tm_map(train_corpus,stemDocument)

train_dtm <- DocumentTermMatrix(train_corpus)
#train_dtm

train_dtm <- removeSparseTerms(train_dtm,0.995)

trainsparse <- as.data.frame(as.matrix(train_dtm))
colnames(trainsparse) <- make.names(colnames(trainsparse))
head(trainsparse)

trainsparse$Neu <- train$Neu

set.seed(123) 
library(caTools)
spl <- sample.split(trainsparse$Neu,SplitRatio=0.7) 
train <- subset(trainsparse,spl==TRUE);             
valid <- subset(trainsparse,spl==FALSE);             

model1 <- glm(Neu~.,data=train,family=binomial)
summary(model1)

predict1 <- predict(model1,newdata=train,type="response")
predict2 <- predict(model1,newdata=valid,type="response")

accuracy <- function(predict_object, data, threshold=0.5) {
  return(sum(diag(table(predict_object >= threshold, data))) /
           sum(table(predict_object >= threshold, data)))
}

accuracy(predict1, train$Neu)
accuracy(predict2, valid$Neu)

# The accuracy of logistic regression model in the train data is 0.7013239
# The accuracy of logistic regression model in the validation data is 0.6813844

# Predicting on test data
test <- read.csv("test.csv")

library(tm)
test_corpus <- Corpus(VectorSource(test$tweet))

# Data pre-processing of test corpus
test_corpus <- tm_map(test_corpus, content_transformer(tolower))
test_corpus <- tm_map(test_corpus,removeWords,stopwords("english"))
test_corpus <- tm_map(test_corpus,removePunctuation)
library(SnowballC)
test_corpus <- tm_map(test_corpus,stemDocument)

test_dtm <- DocumentTermMatrix(test_corpus)

test_dtm <- removeSparseTerms(test_dtm,0.995)

new_test <- as.data.frame(as.matrix(test_dtm))
colnames(new_test) <- make.names(colnames(new_test))

library(tidyverse)

common_colnames <- intersect(names(trainsparse), names(new_test))
new_test <- select(new_test, common_colnames)
new_train <- select(trainsparse, common_colnames)
new_train$Neu <- trainsparse$Neu

model2 <- glm(Neu~.,data=new_train,family=binomial)

Neu_predict <- predict(model2,newdata=new_test,type="response")

Neu_predict <- as.data.frame(Neu_predict)
Neu_predict$Neu_binary <- ifelse(Neu_predict$Neu_predict >= 0.5, 2, 0)
table(Neu_predict$Neu_binary)
head(Neu_predict)

################################################################################
# NEGATIVES PREDICTION
################################################################################

train <- read.csv("train.csv",stringsAsFactors=FALSE)

train$Neg <- as.factor(train$sentiment==1)
train$Neg <- ifelse(train$Neg == TRUE, 1, 0)

library(tm)
train_corpus <- Corpus(VectorSource(train$tweet))

# Data pre-processing of train corpus
train_corpus <- tm_map(train_corpus, content_transformer(tolower))
train_corpus <- tm_map(train_corpus,removeWords,stopwords("english"))
train_corpus <- tm_map(train_corpus,removePunctuation)
library(SnowballC)
train_corpus <- tm_map(train_corpus,stemDocument)

train_dtm <- DocumentTermMatrix(train_corpus)
#train_dtm

train_dtm <- removeSparseTerms(train_dtm,0.995)

trainsparse <- as.data.frame(as.matrix(train_dtm))
colnames(trainsparse) <- make.names(colnames(trainsparse))
head(trainsparse)

trainsparse$Neg <- train$Neg

set.seed(123) 
library(caTools)
spl <- sample.split(trainsparse$Neg,SplitRatio=0.7) 
train <- subset(trainsparse,spl==TRUE);             
valid <- subset(trainsparse,spl==FALSE);             

model1 <- glm(Neg~.,data=train,family=binomial)
summary(model1)

predict1 <- predict(model1,newdata=train,type="response")
predict2 <- predict(model1,newdata=valid,type="response")

accuracy <- function(predict_object, data, threshold=0.5) {
  return(sum(diag(table(predict_object >= threshold, data))) /
           sum(table(predict_object >= threshold, data)))
}

accuracy(predict1, train$Neg)
accuracy(predict2, valid$Neg)

# The accuracy of logistic regression model in the train data is 0.8087614
# The accuracy of logistic regression model in the validation data is 0.8021996

# Predicting on test data
test <- read.csv("test.csv")

library(tm)
test_corpus <- Corpus(VectorSource(test$tweet))

# Data pre-processing of test corpus
test_corpus <- tm_map(test_corpus, content_transformer(tolower))
test_corpus <- tm_map(test_corpus,removeWords,stopwords("english"))
test_corpus <- tm_map(test_corpus,removePunctuation)
library(SnowballC)
test_corpus <- tm_map(test_corpus,stemDocument)

test_dtm <- DocumentTermMatrix(test_corpus)

test_dtm <- removeSparseTerms(test_dtm,0.995)

new_test <- as.data.frame(as.matrix(test_dtm))
colnames(new_test) <- make.names(colnames(new_test))

library(tidyverse)

common_colnames <- intersect(names(trainsparse), names(new_test))
new_test <- select(new_test, common_colnames)
new_train <- select(trainsparse, common_colnames)
new_train$Neg <- trainsparse$Neg

model2 <- glm(Neg~.,data=new_train,family=binomial)

Neg_predict <- predict(model2,newdata=new_test,type="response")

Neg_predict <- as.data.frame(Neg_predict)
Neg_predict$Neg_binary <- ifelse(Neg_predict$Neg_predict >= 0.5, 1, 0)
table(Neg_predict$Neg_binary)
head(Neg_predict)

################################################################################

# Preparing a data frame containing the predicted values 

logreg_predictions <- 1:6871
logreg_predictions <- as.data.frame(logreg_predictions)

logreg_predictions$Pos_Prob <- Pos_predict$Pos_predict
logreg_predictions$Neu_Prob <- Neu_predict$Neu_predict
logreg_predictions$Neg_Prob <- Neg_predict$Neg_predict

# Comparing the predicted probabilities of the general sentiment of test tweet entries being positive, negative and neutral
logreg_predictions$sentiment_3 <- ifelse(logreg_predictions$Pos_Prob > logreg_predictions$Neg_Prob & logreg_predictions$Pos_Prob > logreg_predictions$Neu_Prob, 3, 0)
logreg_predictions$sentiment_2 <- ifelse(logreg_predictions$Neu_Prob > logreg_predictions$Neg_Prob & logreg_predictions$Neu_Prob > logreg_predictions$Pos_Prob, 2, 0)
logreg_predictions$sentiment_1 <- ifelse(logreg_predictions$Neg_Prob > logreg_predictions$Neu_Prob & logreg_predictions$Neg_Prob > logreg_predictions$Pos_Prob, 1, 0)

logreg_predictions$sentiment <- logreg_predictions$sentiment_3 + logreg_predictions$sentiment_2 + logreg_predictions$sentiment_1
head(logreg_predictions)
table(logreg_predictions$sentiment)

# Data manipulation for exporting finalised csv file 

logreg_predictions <- select(logreg_predictions, c(logreg_predictions, sentiment))
head(logreg_predictions)

library(dplyr)
logreg_predictions <- logreg_predictions %>% rename(id = logreg_predictions)

write.csv(logreg_predictions,"logreg_predictions.csv", row.names = FALSE)
