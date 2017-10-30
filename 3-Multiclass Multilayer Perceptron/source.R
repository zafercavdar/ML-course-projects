library(MASS)

set.seed(521)
# mean parameters
class_means <- matrix(c(+2.0, +2.0,
                        -4.0, -4.0,
                        -2.0, +2.0,
                        +4.0, -4.0,
                        -2.0, -2.0,
                        +4.0, +4.0,
                        +2.0, -2.0,
                        -4.0, +4.0), 2, 8)
# covariance parameters
class_covariances <- array(c(+0.8, -0.6, -0.6, +0.8,
                             +0.4, +0.0, +0.0, +0.4,
                             +0.8, +0.6, +0.6, +0.8,
                             +0.4, +0.0, +0.0, +0.4,
                             +0.8, -0.6, -0.6, +0.8,
                             +0.4, +0.0, +0.0, +0.4,
                             +0.8, +0.6, +0.6, +0.8,
                             +0.4, +0.0, +0.0, +0.4), c(2, 2, 8))
# sample sizes
class_sizes <- rep(50, 8)

# generate random samples
points1 <- mvrnorm(n = class_sizes[1], mu = class_means[,1], Sigma = class_covariances[,,1])
points2 <- mvrnorm(n = class_sizes[2], mu = class_means[,2], Sigma = class_covariances[,,2])
points3 <- mvrnorm(n = class_sizes[3], mu = class_means[,3], Sigma = class_covariances[,,3])
points4 <- mvrnorm(n = class_sizes[4], mu = class_means[,4], Sigma = class_covariances[,,4])
points5 <- mvrnorm(n = class_sizes[5], mu = class_means[,5], Sigma = class_covariances[,,5])
points6 <- mvrnorm(n = class_sizes[6], mu = class_means[,6], Sigma = class_covariances[,,6])
points7 <- mvrnorm(n = class_sizes[7], mu = class_means[,7], Sigma = class_covariances[,,7])
points8 <- mvrnorm(n = class_sizes[8], mu = class_means[,8], Sigma = class_covariances[,,8])
X <- rbind(points1, points2, points3, points4, points5, points6, points7, points8)
colnames(X) <- c("x1", "x2")

# generate corresponding labels
y <- c(rep(1, class_sizes[1]),rep(1, class_sizes[2]),rep(2, class_sizes[3]),rep(2, class_sizes[4]),
       rep(3, class_sizes[5]),rep(3, class_sizes[6]),rep(4, class_sizes[7]),rep(4, class_sizes[8]))
y_truth <- y

# get number of samples and number of features
N <- length(y_truth)
D <- ncol(X)
K <- max(y_truth)

y_truth_encoded <- t(sapply(1:N, FUN = function(i) {
  if (y[i] == 1) {
    return(c(1,0,0,0))
  } else if (y[i] == 2) {
    return(c(0,1,0,0))
  } else if (y[i] == 3) {
    return(c(0,0,1,0))
  } else if (y[i] == 4) {
    return(c(0,0,0,1))
  }
}))

# plot data points generated
plot(points1[,1], points1[,2], type = "p", pch = 19, col = "red", las = 1,
     xlim = c(-6, 6), ylim = c(-6, 6),
     xlab = "x1", ylab = "x2")
points(points2[,1], points2[,2], type = "p", pch = 19, col = "red")
points(points3[,1], points3[,2], type = "p", pch = 19, col = "green")
points(points4[,1], points4[,2], type = "p", pch = 19, col = "green")
points(points5[,1], points5[,2], type = "p", pch = 19, col = "blue")
points(points6[,1], points6[,2], type = "p", pch = 19, col = "blue")
points(points7[,1], points7[,2], type = "p", pch = 19, col = "magenta")
points(points8[,1], points8[,2], type = "p", pch = 19, col = "magenta")

# define the sigmoid function
sigmoid <- function(a) {
  return (1 / (1 + exp(-a)))
}

# define the softmax function
softmax <- function(Z, v) {
  y_predict_not_normalized <- exp(cbind(1, Z) %*% v)
  y_predict_normalized <- matrix(0, nrow=nrow(y_predict_not_normalized), ncol=ncol(y_predict_not_normalized))
  for (i in 1:nrow(y_predict_not_normalized)) {
    y_predict_normalized[i,] <- y_predict_not_normalized[i,] / sum(y_predict_not_normalized[i,])
  }
  return(y_predict_normalized)
}

getLabels <- function(y_predicted) {
  sapply(1:N, FUN = function(i) {
    which.max(y_predicted[i,])
  })
}

H <- 20

W <- matrix(runif((D + 1) * H, min = -0.01, max = 0.01), D + 1, H)
v <- matrix(runif((H + 1) * K, min = -0.01, max = 0.01), H + 1, K)

Z <- sigmoid(cbind(1, X) %*% W)
y_predicted <- softmax(Z,v)

# update this function
objective_values <- -sum(y_truth_encoded * log(y_predicted + 1e-100))

# set learning parameters
eta <- 0.1 # learning rate
epsilon <- 1e-3
max_iteration <- 200

# learn W and v using gradient descent and online learning
iteration <- 1
while (1) {
  for (i in sample(N)) {
    # calculate hidden nodes
    Z[i,] <- sigmoid(c(1, X[i,]) %*% W)
    # calculate output node
    y_predicted[i,] <- softmax(matrix(Z[i,], 1, H), v)

    delta_v <- matrix(0, nrow=H+1, ncol=K)
    for (k in 1:K) {
      # v[, k] <- v[, k] + eta * (y_truth_encoded[i, k] - y_predicted[i, k]) * c(1, Z[i,]) 
      delta_v[, k] <- (y_truth_encoded[i, k] - y_predicted[i, k]) * c(1, Z[i,])
    }
    
    delta_W <- matrix(0, nrow=D+1, ncol=H)
    for (h in 1:H) {
      inner_term = sum((y_truth_encoded[i,] - y_predicted[i,]) * v[h,])
      delta_W[, h] <- inner_term * Z[i, h] * (1 - Z[i, h]) * c(1, X[i,])
      #W[, h] <- W[, h] + eta * inner_term * Z[i, h] * (1 - Z[i, h]) * c(1, X[i,])
    }
    
    v <- v + eta * delta_v
    W <- W + eta * delta_W
  }
  
  Z <- sigmoid(cbind(1, X) %*% W)
  y_predicted <- softmax(Z, v)
  objective_values <- c(objective_values, -sum(y_truth_encoded * log(y_predicted + 1e-100)))
  
  if (abs(objective_values[iteration + 1] - objective_values[iteration]) < epsilon | iteration >= max_iteration) {
    break
  }
  
  iteration <- iteration + 1
  print(iteration)
}
print(W)
print(v)

# plot objective function during iterations
plot(1:(iteration + 1), objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

# calculate confusion matrix
y_predicted_decoded <- getLabels(y_predicted)
confusion_matrix <- table(y_predicted_decoded, y_truth)
print(confusion_matrix)

# define score function for arbitrary points
score_function <- function(x1,x2) {
  temp_Z <- sigmoid(cbind(1, x1, x2) %*% W)
  single_y_predicted <- softmax(temp_Z, v)
  return(which.max(single_y_predicted))
}

# evaluate discriminat function on a grid
x1_interval <- seq(from = -6, to = +6, by = 0.06)
x2_interval <- seq(from = -6, to = +6, by = 0.06)
x1_grid <- matrix(x1_interval, nrow = length(x1_interval), ncol = length(x1_interval), byrow = FALSE)
x2_grid <- matrix(x2_interval, nrow = length(x2_interval), ncol = length(x2_interval), byrow = TRUE)

predicted_labels <- matrix(mapply(score_function, x1_grid, x2_grid), nrow(x2_grid), ncol(x2_grid))

plot(X[y_truth == 1, 1], X[y_truth == 1, 2], type = "p", pch = 19, col = "red",
     xlim = c(-6, +6),
     ylim = c(-6, +6),
     xlab = "x1", ylab = "x2", las = 1)
points(X[y_truth == 2, 1], X[y_truth == 2, 2], type = "p", pch = 19, col = "green")
points(X[y_truth == 3, 1], X[y_truth == 3, 2], type = "p", pch = 19, col = "blue")
points(X[y_truth == 4, 1], X[y_truth == 4, 2], type = "p", pch = 19, col = "magenta")

# wrong predictions
points(X[y_predicted_decoded != y_truth, 1], X[y_predicted_decoded != y_truth, 2], cex = 1.5, lwd = 2)

# grid
alpha = 0.05
points(x1_grid[predicted_labels == 1], x2_grid[predicted_labels == 1], col = rgb(red = 1, green = 0, blue = 0, alpha = alpha), pch = 16)
points(x1_grid[predicted_labels == 2], x2_grid[predicted_labels == 2], col = rgb(red = 0, green = 1, blue = 0, alpha = alpha), pch = 16)
points(x1_grid[predicted_labels == 3], x2_grid[predicted_labels == 3], col = rgb(red = 0, green = 0, blue = 1, alpha = alpha), pch = 16)
points(x1_grid[predicted_labels == 4], x2_grid[predicted_labels == 4], col = rgb(red = 1, green = 0, blue = 1, alpha = alpha), pch = 16)
