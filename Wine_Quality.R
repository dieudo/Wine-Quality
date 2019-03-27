#########Dieudonne Ouedraogo   03-22-2019 ################################################

########## Wine  quality predictions #################################################

#Load Libraries 
library(caret)
library(corrplot)
library(tibble)
library(plsRglm)
library(Hmisc)
library(tidyverse)
library(moments)
set.seed(123)
#In case the files don't exist in the directory, they will be downloaded.
##
fileName <- 'winequality-red.csv';
if (!file.exists(fileName)) {
  download.file(paste0('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/', fileName), fileName, method="curl")
}
fileName <- 'winequality-white.csv';
if (!file.exists(fileName)) {
  download.file(paste0('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/', fileName), fileName, method="curl")
}

white <- read.csv('winequality-white.csv', sep=';')
red <- read.csv('winequality-red.csv', sep=';')
str(red)
str(white)
data <- rbind(white,red)
str(data)

#Summary of the dataset
summary(data$quality)
table(data$quality)

###Plot of the target varaible 'quality'
ggplot(data = data) +
  geom_bar(mapping = aes(x = quality),colour="blue")+
  ggtitle("Quality Distribution")



#Correlation plots
corrplot(cor(data),method="number")
## More information about the data 
describe(data)


## Split the dataset into training and testing with 80% training and 20% testing
partition <- createDataPartition(data$quality, p = 0.80)[[1]]
train <- data[partition,]
test <- data[-partition,]
#test

#################################### Model without preprocessing #########################################
### Fit a model by using the training dataset output 'quality' versus all other variables 
### With plsRglm library as method through caret


#10 fold crossvalidation
train_control <- trainControl(method="cv", number=10)
#Setting model specific parameters using grid search
grid <- expand.grid(.nt=length(train)-1, .alpha.pvals.expli=.05)
#Training
model1 <- train(quality~., data=train, trControl=train_control, method="plsRglm", tuneGrid=grid, verbose=FALSE)
str(model1)
##Get the predicted results using the testing dataset
pred1<-predict(model1,newdata = test)
##Compute the mean absolute error between 
##predicted values and actual values using the testing dataset
mean(abs(pred1-test$quality))#0.5529662  

###Since quality are all integers numbers, we could round the results 
pred_round<- round(predict(model1, newdata=test)) ##
#### MAE  on rounded results 
mean(abs(pred_round-test$quality))##0.5100154 




################################ Model with scaled and centered data #############################

Model2 <- train(quality ~ ., data = train, method = 'plsRglm', preProcess = c("center", "scale"),trControl=train_control, tuneGrid=grid, verbose=FALSE)

pred2<-predict(Model2,newdata = test)
#pred2
##Compute the mean absolute error between 
##predicted values and actual values using the testing dataset
mean(abs(pred2-test$quality)) ## 0.5529662




############################# Models using Manual Processing of Data ######################################


#Function for data processing and features Engineering 

data_prep <- function(data){
  #Feature engineering
  #From the correlation table total sulfur and free sulfur are correlated
  #Build a feature which is the ratio of the 2 variables
  data <- add_column(data, sulfur.ratio = data$free.sulfur.dioxide/data$total.sulfur.dioxide, .before = 'quality')
  
  #Feature selection. Dropping redundant variables.
  data$total.sulfur.dioxide <- data$free.sulfur.dioxide <- data$density <- NULL
  
  #Normalizing data using log1p transformation.
  log_data <- log1p(data)
  log_data$quality <- data$quality
  
  #Normalizing variable by subtracting mean and taking the absolute value.
  log_data$residual.sugar <- abs(log_data$residual.sugar - mean(log_data$residual.sugar))
  log_data
}

#Function for outlier detection and removal
remove_outliers <- function(data){
  #Standardizing to mean 0
  scaled_data <- data.frame(scale(data))
  scaled_data$quality <- data$quality
  outliers <- data.frame()
  #Looping over data for outliers
  for (i in 1:nrow(scaled_data)){
    for (x in scaled_data[i,1:length(scaled_data)-1]){
      if (x > 4 | x < -4){
        outliers <- rbind(outliers, i)
      }
    }
  }
  outliers <- data.frame(table(outliers))
  L <- outliers$Freq > 0
  indexs <- array(outliers$outliers[L])
  #Dropping rows with outliers
  no_outliers <- data[!rownames(data) %in% indexs, ]
  no_outliers
}




#Setting training parameters

# 10 fold cross-validation with with grid search 
train_control <- trainControl(method="cv", number=10)
#Setting model specific parameters
grid <- expand.grid(.nt=length(data_prep(train))-1, .alpha.pvals.expli=.05)
#Training
model <- train(quality~., data=data_prep(train), trControl=train_control, method="plsRglm", tuneGrid=grid, verbose=FALSE)

#Predicting
pred <- predict(model, newdata = data_prep(test))
#Return MAE
MAE(pred, data_prep(test)$quality)      #0.559248

model_no_outlier <- train(quality~., data=remove_outliers(train), trControl=train_control, method="plsRglm", tuneGrid=grid, verbose=FALSE)
pred_no_outlier <- predict(model_no_outlier, newdata = remove_outliers(test))
#Return MAE
MAE(pred_no_outlier, remove_outliers(test)$quality) #0.5529642





################################## Models for each dataset#########
##Instead using a combined model on both wine type,
### we could build two models on each wine type 

partition_red <- createDataPartition(red$quality, p = 0.80)[[1]]
train_red <- red[partition_red,]
test_red <- red[-partition_red,]

fit_red <- train(quality ~ ., data = train_red, method = 'plsRglm', tuneGrid=grid, verbose=FALSE,trcontrol=train_control)

results_red<-predict(fit_red,newdata = test_red)
mean(abs(results_red-test_red$quality))#0.4828223


####White wine 
partition_white <- createDataPartition(white$quality, p = 0.80)[[1]]
train_white <- white[partition_white,]
test_white <- white[-partition_white,]

fit_white <- train(quality ~ ., data = train_white, method = 'plsRglm',tuneGrid=grid,verbose=FALSE,trcontrol=train_control)

results_white<-predict(fit_white,newdata = test_white)
mean(abs(results_white-test_white$quality)) #0.587463
#The white wine is more difficult to predict


#########################################################################################################






##########################BEST MODEL RETAINED ############################################################




# 10 fold cross-validation with with grid search 
train_control <- trainControl(method="cv", number=10)
#Setting model specific parameters
grid <- expand.grid(.nt=length(data_prep(train))-1, .alpha.pvals.expli=.05)

#Function for outlier detection and removal
remove_outliers <- function(data){
  #Standardizing to mean 0
  scaled_data <- data.frame(scale(data))
  scaled_data$quality <- data$quality
  outliers <- data.frame()
  #Looping over data for outliers
  for (i in 1:nrow(scaled_data)){
    for (x in scaled_data[i,1:length(scaled_data)-1]){
      if (x > 4 | x < -4){
        outliers <- rbind(outliers, i)
      }
    }
  }
  outliers <- data.frame(table(outliers))
  L <- outliers$Freq > 0
  indexs <- array(outliers$outliers[L])
  #Dropping rows with outliers
  no_outliers <- data[!rownames(data) %in% indexs, ]
  no_outliers
}


model_no_outlier <- train(quality~., data=remove_outliers(train), trControl=train_control, method="plsRglm", tuneGrid=grid, verbose=FALSE)
pred_no_outlier <- predict(model_no_outlier, newdata = remove_outliers(test))
#Return MAE
MAE(pred_no_outlier, remove_outliers(test)$quality) #0.5529642

############################################  END  #########################################

