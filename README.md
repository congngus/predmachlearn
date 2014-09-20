
## 1. Loading and preprocessing the data.

### Downloading the data if need.

```r
## Checks "data" directory if it doesn't exist then create.
if (!file.exists("data")) {
  message("Creating data directory")
  dir.create("data")
}

## Check is training data dowloaded before.
if (!file.exists("data/pml-training.csv")) {
  ## Data's URL
  trainingFileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
  
  ## File's destination.
  dataFile ="data/pml-training.csv"

  ## Download data.
  download.file(trainingFileURL, destfile=dataFile, method="curl")
}
```

### Data overview.

```r
trainingDataSet <- read.csv(file="data/pml-training.csv",head=TRUE,sep=",")
```


```r
dim(trainingDataSet)
```

```
## [1] 19622   160
```


```r
table(trainingDataSet$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

## 2. Create Cross-validation data: 60% training, 40% testing.

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
trainset <- createDataPartition(trainingDataSet$classe, p = 0.6, list = FALSE)
trainingData <- trainingDataSet[trainset, ]
testingData <- trainingDataSet[-trainset, ]

dim(trainingData)
```

```
## [1] 11776   160
```

```r
dim(testingData)
```

```
## [1] 7846  160
```

### Cleaning data support for prediction performance.

```r
# remove near zero variance collumn.
trainingData <- trainingData[, -nearZeroVar(trainingData)]
testingData <- testingData[, -nearZeroVar(testingData)]


## remove columns have more than 20% missing values
totalNA <- sapply(trainingData, function(x) {
    sum(!(is.na(x) | x == ""))
})

NAColumn <- names(totalNA[totalNA < 0.2 * length(trainingData$classe)])

## Remove columns are not importance in predicting Classe variable in dataset.
unuseColumn <- c("X", "user_name", "timestamp", "new_window")
removeColumns <- c(unuseColumn, NAColumn)

trainingData <- trainingData[, !names(trainingData) %in% removeColumns]
testingData <- testingData[, !names(testingData) %in% removeColumns]
```

## 3.Predict with trees. Using random forest as prediction agorithm.
Why I chose Random forests ?
- Random forests are usually one of the two top performing algorithms along with boosting in prediction contests.
- Random forests are very accurate.

```r
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
rfModel <- randomForest(classe ~ ., data = trainingData, importance = TRUE, ntrees = 10)
```

### Random forest model performance on training data.

```r
prdTraining <- predict(rfModel, trainingData)
print(confusionMatrix(prdTraining, trainingData$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3348    0    0    0    0
##          B    0 2279    0    0    0
##          C    0    0 2054    0    0
##          D    0    0    0 1930    0
##          E    0    0    0    0 2165
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

### Random forest model performance on testing data (Out-of-Sample).

```r
prdCrossValidation <- predict(rfModel, testingData)
print(confusionMatrix(prdCrossValidation, testingData$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    1    0    0    0
##          B    0 1517    5    0    0
##          C    0    0 1360    1    0
##          D    0    0    3 1285    5
##          E    0    0    0    0 1437
## 
## Overall Statistics
##                                         
##                Accuracy : 0.998         
##                  95% CI : (0.997, 0.999)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.998         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.999    0.994    0.999    0.997
## Specificity             1.000    0.999    1.000    0.999    1.000
## Pos Pred Value          1.000    0.997    0.999    0.994    1.000
## Neg Pred Value          1.000    1.000    0.999    1.000    0.999
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.173    0.164    0.183
## Detection Prevalence    0.285    0.194    0.173    0.165    0.183
## Balanced Accuracy       1.000    0.999    0.997    0.999    0.998
```
Out-of-sample error is 1 - 0.998 = 0.002 so model have done well.


