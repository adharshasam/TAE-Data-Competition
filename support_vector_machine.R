train <- read.csv("train.csv", stringsAsFactors = FALSE)

library(tm)

train_corpus <- Corpus(VectorSource(train$tweet))

# Data pre-processing of train corpus
train_corpus <- tm_map(train_corpus, content_transformer(tolower))
train_corpus <- tm_map(train_corpus,removeWords,stopwords("english"))
train_corpus <- tm_map(train_corpus,removePunctuation)

library(SnowballC)

train_corpus <- tm_map(train_corpus,stemDocument)
train_dtm <- DocumentTermMatrix(train_corpus)
train_dtm <- removeSparseTerms(train_dtm,0.995)
trainsparse <- as.data.frame(as.matrix(train_dtm))
colnames(trainsparse) <- make.names(colnames(trainsparse))
trainsparse$sentiment <- train$sentiment

# Predicting on test data
test <- read.csv("test.csv")

test_corpus <- Corpus(VectorSource(test$tweet))

# Data pre-processing of test corpus
test_corpus <- tm_map(test_corpus, content_transformer(tolower))
test_corpus <- tm_map(test_corpus,removeWords,stopwords("english"))
test_corpus <- tm_map(test_corpus,removePunctuation)
test_corpus <- tm_map(test_corpus,stemDocument)

test_dtm <- DocumentTermMatrix(test_corpus)

test_dtm <- removeSparseTerms(test_dtm,0.995)

new_test <- as.data.frame(as.matrix(test_dtm))
colnames(new_test) <- make.names(colnames(new_test))

library(tidyverse)
common_colnames <- intersect(names(trainsparse), names(new_test))
new_test <- select(new_test, common_colnames)
new_train <- select(trainsparse, common_colnames)
new_train$sentiment <- trainsparse$sentiment

# SVM
set.seed(123)
library(e1071)
svm_classifier <- svm(as.factor(sentiment) ~ ., new_train, type = 'C-classification', kernel = 'linear', scale = TRUE, probability = TRUE)
svm_predictions <- predict(svm_classifier, newdata = new_test, decision.values = TRUE, probability = TRUE)
probabilities <- attr(svm_predictions, "probabilities")
svm_probabilities <- as.data.frame(probabilities) 

# Preparing a data frame of SVM predictions to export as csv file 
svm_predictions <- 1:6871
svm_predictions <- as.data.frame(svm_predictions)

svm_predictions$Pos_Prob <- svm_probabilities$`3`
svm_predictions$Neu_Prob <- svm_probabilities$`2`
svm_predictions$Neg_Prob <- svm_probabilities$`1`

svm_predictions$sentiment_3 <- ifelse(svm_predictions$Pos_Prob > svm_predictions$Neg_Prob & svm_predictions$Pos_Prob > svm_predictions$Neu_Prob, 3, 0)
svm_predictions$sentiment_2 <- ifelse(svm_predictions$Neu_Prob > svm_predictions$Neg_Prob & svm_predictions$Neu_Prob > svm_predictions$Pos_Prob, 2, 0)
svm_predictions$sentiment_1 <- ifelse(svm_predictions$Neg_Prob > svm_predictions$Neu_Prob & svm_predictions$Neg_Prob > svm_predictions$Pos_Prob, 1, 0)

svm_predictions$sentiment <- svm_predictions$sentiment_3 + svm_predictions$sentiment_2 + svm_predictions$sentiment_1
head(svm_predictions)
table(svm_predictions$sentiment)

# Data manipulation for exporting finalised csv file 
svm_predictions <- select(svm_predictions, c(svm_predictions, sentiment))
head(svm_predictions) # checking!

library(dplyr)
svm_predictions <- svm_predictions %>% rename(id = svm_predictions)

write.csv(svm_predictions,"svm_predictions.csv", row.names = FALSE)