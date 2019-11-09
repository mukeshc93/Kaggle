library(xgboost)
library(dplyr)
library(readr)
library(doParallel)
library(foreach)
#train=read_csv("~/R/My Projects/facebook/train/train.csv")
#test=read_csv("~/R/My Projects/facebook/test/test.csv")


numcl=(count(distinct(as.data.frame(train$place_id)))$n)
y=cbind(levels(as.factor(train$place_id)),(0:(numcl-1)));colnames(y)=c("place_id","level")
train$place_id=as.factor(train$place_id)
#train=merge(train,y,by.x="place_id",all=TRUE)
####write.csv(train,"modtrain.csv",row.names = F)
#train$place_id=NULL
train$row_id=NULL

train=read_csv("~/R/My Projects/facebook/modtrain.csv");train$row_id=NULL
train1=as.data.frame(train[which(train$x<0.5 & train$y<0.2),])
numcl1=(count(distinct(as.data.frame(train1$place_id)))$n)
y1=cbind(levels(as.factor(train1$place_id)),(0:(numcl1-1)));colnames(y1)=c("place_id","level")
train1=merge(train1,y1,by.x="place_id")
z=train1$level
rm(train)
train1$place_id=NULL

train1=xgb.DMatrix(as.matrix(train1[,-ncol(train1)]), label=(as.numeric(z)-1))


params=list("eta"=0.3,"subsample"=0.8,objective="multi:softmax","num_class"=numcl1)
ml1=xgb.train(params,train1,nrounds = 20,print.every.n = 1, nthread=4)
xgb.save(ml1,"ml1.model")


###########ML
x=cbind(rep(seq(0.2,10,0.2),20),sort(rep(seq(0.5,10,0.5),50)))
train=read_csv("~/R/My Projects/facebook/modtrain.csv");train$row_id=NULL
train$time=((train$time/60) %%24);train$time=round(train$time,2)
train$x=round(train$x,3);train$y=round(train$y,3)

#cl <- makeCluster(3)
#registerDoParallel(cl)

for( i in 1:1000)
{
train1=as.data.frame(train[which(train$x<(x[i,2]) & (train$y<x[i,1] & train$y>(x[i,1]-0.2))),])
numcl1=(count(distinct(as.data.frame(train1$place_id)))$n)
y1=cbind(levels(as.factor(train1$place_id)),(0:(numcl1-1)));colnames(y1)=c("place_id","level")
train1=merge(train1,y1,by.x="place_id")
z=train1$level
train1$place_id=NULL
print(paste("nrow for for iteration no. ",i,"is",nrow(train1),"and no. of labels=",numcl1))
train1=xgb.DMatrix(as.matrix(train1[,-ncol(train1)]), label=(as.numeric(z)-1))

params=list("eta"=0.7,"subsample"=0.8,objective="multi:softmax","num_class"=numcl1)
ml1=xgb.train(params,train1,nrounds = 10,print.every.n = 1, nthread=4)
xgb.save(ml1,paste("ml",i,".model",sep=""))
}
#stopCluster(cl)
