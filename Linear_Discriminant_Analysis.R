library(mda)
install.packages("mda")
Predict_Gender<-read.csv(file = "C:/Users/Aparna/Documents/TrainingSet_lda.csv",header = TRUE, sep = ",")
summary(Predict_Gender)
unique(Predict_Gender$Gender)
table(Predict_Gender$Gender)
cor(Predict_Gender[,1:3])
split.screen(c(1,2))
screen(1)
boxplot(Height~Gender,data=Predict_Gender,col="lavender")
boxplot(Weight~Gender,data=Predict_Gender,col="lavender")
boxplot(Age~Gender,data=Predict_Gender,col="lavender")
close.screen(all=TRUE)
model1<- prcomp(Predict_Gender[,1:3],scale = TRUE )
biplot(model1)
summary(model1)

model2<- mda(Gender ~., data=Predict_Gender)
plot(model2)
table(Predict_Gender$Gender,predict(model2))
tbl<-table(Predict_Gender$Gender,predict(model2))
sum(diag(tbl))/sum(tbl)

Predict_Gender[1,]
predict(model2, Predict_Gender[1,])

Test_Data<- Predict_Gender[1,]
Test_Data

#Run this part of code line by line to accept the input from the user
Entered_height<-readline(prompt = "Enter height:")
Entered_weight<-readline(prompt = "Enter weight:")
Entered_age<-readline(prompt = "Enter age:")
Entered_height<-as.integer(Entered_height)
Entered_weight<-as.integer(Entered_weight)
Entered_age<-as.integer(Entered_age)

#Run this to test the accept input from the user
Test_Data[1,]<-data.frame(Entered_height,Entered_weight,Entered_age,'Unknown')
Test_Data
Test_Data<-rbind(Test_Data)
predict(model2,Test_Data)
class<-predict(model2,Test_Data)
class
