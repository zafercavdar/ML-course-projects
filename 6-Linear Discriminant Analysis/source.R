# COMP 421 Fall 2017 Koc University
# HW06- Zafer Cavdar - 0049995

training_digits <- read.csv("hw06_mnist_training_digits.csv", header = FALSE)
training_labels <- read.csv("hw06_mnist_training_labels.csv", header = FALSE)
test_digits <- read.csv("hw06_mnist_test_digits.csv", header = FALSE)
test_labels <- read.csv("hw06_mnist_test_labels.csv", header = FALSE)

# get X and y values
X_train <- as.matrix(training_digits) / 255
X_test <- as.matrix(test_digits) / 255
y_train <- training_labels[,1]
y_test <- test_labels[,1]

# get number of samples and number of features from training set
N <- length(y_train)
D <- ncol(X_train)
K <- max(y_train)

# calculate means
class_means <- t(sapply(1:K, function(c) {
  colMeans(X_train[y_train == c, ])
}))
X_train_minus_mean <- t((sapply(1:N, function(i) {
  X_train[i, ] - class_means[y_train[i], ]
})))
total_mean <- colMeans(class_means)

# calculate class scatter matrix
Sc <- function (c) {
  res <- matrix(0, D, D)
  for(i in 1:N) {
    if (y_train[i] == c) {
      res <- res + X_train_minus_mean[i, ] %*% t(X_train_minus_mean[i, ])
    }
  }
  return(res)
}

# calculate within class scatter matrix
Sw <- function() {
  sum <- matrix(0, D, D)
  for(c in 1:K) {
    sum <- sum + Sc(c)
  }
  return(sum)
}
SW <- Sw()

# calculate between-class scatter matrix
Sb <- function () {
  sum <- matrix(0, D, D)
  for(c in 1:K) {
    sum <- sum + sum(y_train == c) * (class_means[c,] - total_mean) %*% t(class_means[c,] - total_mean)
  }
  return(sum)
}
SB <- Sb()

# solve the singularity
for (d in 1:D) {
  SW[d,d] <- SW[d,d] + 1e-10
}

# calculate eigenvalues and eigenvectors
SW_inversed <- solve(SW)
solve_this <- SW_inversed %*% SB
decomposition <- eigen(solve_this, symmetric = TRUE)

calcZ <- function (X, R) {
  W <- decomposition$vectors[, 1:R]
  return(X %*% W)
}

Z_train <- calcZ(X_train, 2)
Z_test <- calcZ(X_test, 2)

# plot two-dimensional projections for training and test
point_colors <- c("#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6")
old.par <- par(mfrow=c(1, 2))
plot(Z_train[,1], Z_train[,2], type = "p", pch = 19, col = point_colors[y_train], cex = 0,
     xlab = "Dimension 1", ylab = "Dimension 2", las = 1, main="Training points")
text(Z_train[,1], Z_train[,2], labels = y_train %% 10, col = point_colors[y_train])

plot(Z_test[,1], Z_test[,2], type = "p", pch = 19, col = point_colors[y_test], cex = 0,
     xlab = "Dimension 1", ylab = "Dimension 2", las = 1, main="Test points")
text(Z_test[,1], Z_test[,2], labels = y_test %% 10, col = point_colors[y_test])
par(old.par)

distance <- function(x1, x2) {
  sqrt(sum((x1 - x2) ^ 2)) 
}

# Calculate accuracy for r in 1:9 using 5-nn classifier
accuracy <- c()
for(r in 1:9) {
  Z_train <- calcZ(X_train, r)
  Z_test <- calcZ(X_test, r)
  count <- 0
  for (i in 1:length(Z_test[,1])) {
    distances <- sapply(1:length(Z_train[,1]), function (j) {
      distance(Z_test[i,], Z_train[j,])
    })
    indices <- order(distances)[1:5]
    prediction <- as.numeric(names(which.max(table(y_train[indices]))))
    truth <- y_test[i]
    if (truth == prediction) {
      count <- count + 1
    }
  }
  accuracy[r] <- count
}

# Plot the accuracy vs R
accuracy_percentage <- (accuracy / length(Z_train[,1])) * 100
plot(1:9, accuracy_percentage,
     type = "b", lwd = 2, las = 1, pch = 1,
     xlab = "R", ylab = "Classification accuracy(%)")
