# plots
#test = read.table("log.test")
test = read.csv("err.test")
train = read.table("err.train")

# loss
plot(train$V1,train$V3, type='p', col=4, xlab="Iterations", ylab="Loss", main="VGG two-norm: Train loss vs. iterations")

# accuracy
plot(test$V1, test$V3, type='p', col=4, xlab="Iterations", ylab="Accuracy", main="VGG: Test accuracy vs. iterations")

plot(test$NumIters, test$accuracy, type='p', col=4, xlab="Iterations", ylab="Accuracy", main="VGG two-norm: Test accuracy vs. iterations")
