require(MASS)
set.seed(521)
# mean parameters
class_means <- matrix(c(0, 1.5,
                        -2.5, -3,
                        2.5, -3), 2,3, byrow = FALSE)
class_means_1 <- c(0, 1.5)
class_means_2 <- c(-2.5, -3)
class_means_3 <- c(2.5, 3)
# standard deviation parameters
class_deviations <- array(c(1, 0.2, 0.2, 3.2,
                            1.6, -0.8, -0.8, 1.0,
                            1.6, 0.8, 0.8, 1.0), c(2,2,3)) 
# sample sizes
class_sizes <- c(100, 100, 100)

# generate random samples
points_1 = mvrnorm(class_sizes[1], mu = class_means[,1], Sigma = class_deviations[,,1])
points_2 = mvrnorm(class_sizes[2], mu = class_means[,2], Sigma = class_deviations[,,2])
points_3 = mvrnorm(class_sizes[3], mu = class_means[,3], Sigma = class_deviations[,,3])
X <- rbind(points_1, points_2, points_3)
x1 <- X[,1]
x2 <- X[,2]

# generate corresponding labels
y <- c(rep(1, class_sizes[1]), 
       rep(2, class_sizes[2]),
       rep(3, class_sizes[3]))
y_truth <- y

# plot data points generated
plot(points_1[,1], points_1[,2], type = "p", pch = 19, col = "red", las = 1,
     xlim = c(-6, 6), ylim = c(-6, 6),
     xlab = "x1", ylab = "x2")
points(points_2[,1], points_2[,2], type = "p", pch = 19, col = "green")
points(points_3[,1], points_3[,2], type = "p", pch = 19, col = "blue")


# get number of classes and number of samples
K <- max(y)
N <- length(y)

# define denominator of the sigmoid function
sub_sigmoid <- function(X, W, w0) {
  return (exp(X %*% W + w0))
}

# define the gradient functions
gradient_W <- function(X, y_truth_input, y_predicted_input) {
  sapply(X=1:K, FUN = function(j) {
    rowSums(sapply(X=1:N, FUN = function(t) {
      (y_truth_input[t,j] - y_predicted_input[t,j]) * X[t,]
    }))
  })
}

gradient_w0 <- function(y_truth_input, y_predicted_input) {
  return(colSums(y_truth_input - y_predicted_input))
}

y_truth_encoded <- t(sapply(X = 1:N, FUN = function(n) {
 if (y_truth[n] == 1) {
  return(c(1,0,0)) 
 } else if (y_truth[n] == 2){
   return(c(0,1,0)) 
 } else if (y_truth[n] == 3){
   return(c(0,0,1)) 
 }
}))

# set learning parameters
eta <- 0.01 # step size
epsilon <- 1e-3

# randomly initalize W and w0
W <- sapply(X = 1:K, FUN = function(c) runif(ncol(X), min = -0.01, max = 0.01)) # random uniform
w0 <- runif(K, min = -0.01, max = 0.01) # random uniform

# learn W and w0 using gradient descent
iteration <- 1
objective_values <- c()
while (1) {
  print(paste0("running iteration#", iteration))
  y_predicted_fraction = rowSums(sapply(X = 1:K, FUN = function(d) {
    sub_sigmoid(X, W[,d], w0[d])
  }))
  y_predicted_probs <- sapply(X = 1:K, FUN = function(d) {
    sub_sigmoid(X, W[,d], w0[d]) / y_predicted_fraction
  })
  
  # -loglikelihood = error function
  objective_value <- sum(colSums(sapply(X = 1:N, FUN = function(n) {
    sapply(X= 1:K, FUN = function(c) {
      -1 * y_truth_encoded[n,c] * log(y_predicted_probs[n,c] + epsilon)
    })
  })))
  objective_values <- c(objective_values, objective_value)
  
  W_old <- W
  w0_old <- w0
  
  W <- W + eta * gradient_W(X, y_truth_encoded, y_predicted_probs)
  w0 <- w0 + eta * gradient_w0(y_truth_encoded, y_predicted_probs)
  
  distance_check <- sapply(X=1:K, FUN = function(c) {
    sqrt((w0[c] - w0_old[c])^2 + sum((W[,c] - W_old[,c])^2))
  })
  
  if (sum(distance_check) < K * epsilon / 2) { # euclidean distance
    break
  }
  
  iteration <- iteration + 1
  print(W)
  print(w0)
}
print(W)
print(w0)

# plot objective function during iterations
plot(1:iteration, objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

# calculate confusion matrix
y_predicted <- sapply(X=1:N, FUN = function(n) {
  which.max(y_predicted_probs[n,])
})

confusion_matrix <- table(y_predicted, y_truth)
print(confusion_matrix)

#define prediction function
predict_class <- function(x1, x2) {
  which.max(sapply(X = 1:K, FUN = function(d) {
    sub_sigmoid(array(c(x1, x2), c(1,2)), W[,d], w0[d])
  }))
}

# evaluate discriminat function on a grid
x1_interval <- seq(from = -6, to = +6, by = 0.02)
x2_interval <- seq(from = -6, to = +6, by = 0.02)
x1_grid <- matrix(x1_interval, nrow = length(x1_interval), ncol = length(x1_interval), byrow = FALSE)
x2_grid <- matrix(x2_interval, nrow = length(x2_interval), ncol = length(x2_interval), byrow = TRUE)

predicted_classes <- matrix(mapply(predict_class, x1_grid, x2_grid), nrow(x1_grid), ncol(x2_grid))

#plot the results
plot(x1_grid[predicted_classes == 1], x2_grid[predicted_classes == 1], col = rgb(red = 1, green = 0, blue = 0, alpha = 0.01), pch = 16,
       xlim = c(-6, +6),
       ylim = c(-6, +6),
       xlab = "x1", ylab = "x2", las = 1)
points(x1_grid[predicted_classes == 2], x2_grid[predicted_classes == 2], col = rgb(red = 0, green = 1, blue = 0, alpha = 0.01), pch = 16)
points(x1_grid[predicted_classes == 3], x2_grid[predicted_classes == 3], col = rgb(red = 0, green = 0, blue = 1, alpha = 0.01), pch = 16)
points(X[y_truth == 1, 1], X[y_truth == 1, 2], type = "p", pch = 19, col = "red")
points(X[y_truth == 2, 1], X[y_truth == 2, 2], type = "p", pch = 19, col = "green")
points(X[y_truth == 3, 1], X[y_truth == 3, 2], type = "p", pch = 19, col = "blue")
contour(x1_interval, x2_interval, predicted_classes, levels = c(1,2,3), add = TRUE, lwd = 1, drawlabels = FALSE)
points(X[y_predicted != y_truth, 1], X[y_predicted != y_truth, 2], cex = 1.5, lwd = 2)
