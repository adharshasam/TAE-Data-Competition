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

head(new_train)

# Random Forests
set.seed(123)
library(randomForest)

RF_model <- randomForest(as.factor(sentiment) ~ ., data = new_train)
rf_predictions <- predict(RF_model,newdata = new_test, type="class")
rf_predictions
table(rf_predictions)

write.csv(rf_predictions,"random_forests_predictions.csv", row.names = FALSE)

################################################################################
# The following code is not part of the predictions submitted
################################################################################

vu <- varUsed(RF_model, count = TRUE)
vusorted <- sort(vu, decreasing = FALSE, index.return = TRUE)
dotchart(vusorted$x, names(RF_model$forest$xlevel[vusorted$ix]))
tail(names(RF_model$forest$xlevel[vusorted$ix]), 50)

# Tuning the hyperparameters
# Values of ntree to be tested
B <- c(10,50,100,250,500)
# Values of mtry to be tested
m <- seq(1,20,by=1)
# Initialize a matrix for the OOB value
OOB_matrix <- matrix(0,nrow=length(B),ncol=length(m))
# For loop
for (i in 1:length(B)){
  for (j in 1:length(m)){
    # train a forest with B[i] trees
    forest_temp <- randomForest(as.factor(sentiment) ~ ., data = new_train, ntree = B[i], mtry = m[j])
    # model performance
    OOB_matrix[i,j] <- forest_temp$err.rate[B[i],1]
    # remove the temporary variable
    rm(forest_temp)
  }
}

# Best OOB performance for mtry and ntree
library(reshape2)
rownames(OOB_matrix) <- B
colnames(OOB_matrix) <- m
longData <- melt(OOB_matrix)
#plot
library(ggplot2)
ggplot(longData, aes(x = Var1, y = Var2)) + 
  geom_raster(aes(fill=value)) + 
  scale_fill_gradient(low="grey90", high="red") +
  labs(x="Number of Trees", y="Number of predictors")

# mtry and ntree parameters to be altered based ggplot result of lowest OOB error (grey)
new_RF_model <- randomForest(as.factor(sentiment) ~ ., data = new_train, mtry = 5, ntree = 500)
new_RF_model

new_rf_predictions <- predict(new_RF_model, newdata = new_test, type = "class")
new_rf_predictions

write.csv(new_rf_predictions,"random_forests_predictions.csv", row.names = FALSE)