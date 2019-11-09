library(xgboost)
library(caret)

train = read.csv("file:///C:/Users/mukes_000/Documents/R/My Projects/Kaggle satander cust Satis/train.csv (4)/train.csv", header = T, stringsAsFactors = T)
test = read.csv("file:///C:/Users/mukes_000/Documents/R/My Projects/Kaggle satander cust Satis/test.csv (2)/test.csv", header = T, stringsAsFactors = T)
submision = read.csv("file:///C:/Users/mukes_000/Documents/R/My Projects/Kaggle satander cust Satis/sample_submission.csv (2)/sample_submission.csv", header = T)

id1 = train$ID
id2 = test$ID
train.y = train$TARGET
#train$ID = NULL
#test$ID = NULL
test$TARGET = 2
newd = rbind(train, test)
n = NA
for (i in 1:ncol(train)) {
  if (colMeans(newd)[i] == 0)
    n = rbind(n, i)
}
n = n[!is.na(n)]

newd = newd[, - n]

tar=newd$TARGET
newd$TARGET=NULL

#new = train
#new = test
n = NULL
n1 = NULL
for (i in 1:ncol(newd)) {
  if (class(newd[, i]) == "integer") {
    n[i] = i
  } else if (class(newd[, i]) == "numeric") {
    n1[i] = i
  }
}

n = n[complete.cases(n)]
n1 = n1[complete.cases(n1)]
x = rowMeans(newd[, n], na.rm = T)
x1 = rowMeans(newd[, n1], na.rm = T)
traind = cbind(x, x1, newd)

x = (apply(newd[, n], 1, min))
x1 = (apply(newd[, n], 1, max))
traind = cbind(x, x1, traind)

x = scale(apply(newd[, n1], 1, min))
x1 = scale(apply(newd[, n1], 1, max))
traind = cbind(x, x1, traind)
x = (apply(newd[, n], 1, median))
x1 = (apply(newd[, n], 1, sd))
traind = cbind(x, x1, traind)

#x = (apply(newd[, n1], 1, median))
x1 = (apply(newd[, n1], 1, sd))
traind = cbind(x, x1, traind)

x = which(tar == 2)
test1 = traind[x,]
train1 = traind[-x,]

#train=traind
#test = traind

ml_train = xgb.DMatrix(as.matrix(train1), label = train.y)
ml_test = xgb.DMatrix(as.matrix(test1))
watchlist = list(train = ml_train)
params = list(objective = "binary:logistic",
              eval_metric = "auc", eta = 0.007,
              max_depth = 6,
              min_child_weight = 1,
              subsample = 0.75,
              colsample_bytree = 0.75)
set.seed(143)
ml = xgb.train(params, data = ml_train, nrounds = 3000, verbose = 1, watchlist, maximize = T)
pred1=predict(ml,ml_train)
pred = predict(ml, ml_test)
submision$TARGET = pred
write.csv(submision, "output.csv", row.names = F)