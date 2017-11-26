# Zafer Cavdar - COMP 421 Homework 5 - Decision Tree Regression
# Reference: Textbook Introduction to Machine Learning, Ethem Alpaydin

# QUESTION 1 - Read data, create train and test dataset.
data_set <- read.csv("hw05_data_set.csv")
x_all <- data_set$x
y_all <- data_set$y

set.seed(521)
train_indices <- sample(length(x_all), 100)
x_train <- x_all[train_indices]
y_train <- y_all[train_indices]
x_test <- x_all[-train_indices]
y_test <- y_all[-train_indices]
minimum_value <- floor(min(x_all)) - 2 
maximum_value <- ceiling(max(x_all)) + 2
N_train <- length(x_train)
N_test <- length(x_test)

# QUESTION 2 - Implement Decision Tree Regression algorithm with P pruning parameter

DecisionTreeRegression <- function(P) {
  # reset variables
  node_splits <- c()
  node_means <- c()
  
  # put all training instances into the root node
  node_indices <- list(1:N_train)
  is_terminal <- c(FALSE)
  need_split <- c(TRUE)
  
  # learning algorithm
  while (1) {
    # find nodes that need splitting
    split_nodes <- which(need_split)
    # check whether we reach all terminal nodes
    if (length(split_nodes) == 0) {
      break
    }
    # find best split positions for all nodes
    for (split_node in split_nodes) {
      data_indices <- node_indices[[split_node]]
      need_split[split_node] <- FALSE
      node_mean <- mean(y_train[data_indices])
      if (length(x_train[data_indices]) <= P) {
        is_terminal[split_node] <- TRUE
        node_means[split_node] <- node_mean
      } else {
        is_terminal[split_node] <- FALSE
        unique_values <- sort(unique(x_train[data_indices]))
        split_positions <- (unique_values[-1] + unique_values[-length(unique_values)]) / 2
        split_scores <- rep(0, length(split_positions))
        for (s in 1:length(split_positions)) {
          left_indices <- data_indices[which(x_train[data_indices] <= split_positions[s])]
          right_indices <- data_indices[which(x_train[data_indices] > split_positions[s])]
          total_error <- 0
          if (length(left_indices) > 0) {
            mean <- mean(y_train[left_indices])
            total_error <- total_error + sum((y_train[left_indices] - mean) ^ 2)
          }
          if (length(right_indices) > 0) {
            mean <- mean(y_train[right_indices])
            total_error <- total_error + sum((y_train[right_indices] - mean) ^ 2)
          }
          split_scores[s] <- total_error / (length(left_indices) + length(right_indices))
        }
        if (length(unique_values) == 1) {
          is_terminal[split_node] <- TRUE
          node_means[split_node] <- node_mean
          next 
        }
        best_split <- split_positions[which.min(split_scores)]
        node_splits[split_node] <- best_split
        
        # create left node using the selected split
        left_indices <- data_indices[which(x_train[data_indices] < best_split)]
        node_indices[[2 * split_node]] <- left_indices
        is_terminal[2 * split_node] <- FALSE
        need_split[2 * split_node] <- TRUE
        
        # create right node using the selected split
        right_indices <- data_indices[which(x_train[data_indices] >= best_split)]
        node_indices[[2 * split_node + 1]] <- right_indices
        is_terminal[2 * split_node + 1] <- FALSE
        need_split[2 * split_node + 1] <- TRUE
      }
    }
  }
  result <- list("splits"= node_splits, "means"= node_means, "is_terminal"= is_terminal)
  return(result)
}

# QUESTION 3 - Learn a DT with P = 10 and plot
P <- 10
result <- DecisionTreeRegression(P)
node_splits <- result$splits
node_means <- result$means
is_terminal <- result$is_terminal

# define regression function
get_prediction <- function(dp, is_terminal, node_splits, node_means){
  index <- 1
  while (1) {
    if (is_terminal[index] == TRUE) {
      return(node_means[index])
    } else {
      if (dp <= node_splits[index]) {
        index <- index * 2
      } else {
        index <- index * 2 + 1
      }
    }
  }
}

#plot train data, test data and fit in the figure
plot(x_train, y_train, type = "p", pch = 19, col = "blue",
     ylim = c(min(y_train), max(y_train)), xlim = c(minimum_value, maximum_value),
     ylab = "y", xlab = "x", las = 1)
points(x_test, y_test, type = "p", pch = 19, col= "red")
legend(55,85, legend=c("training", "test"),
       col=c("blue", "red"), pch = 19, cex = 0.5, bty = "y")
grid_interval <- 0.01
data_interval <- seq(from = minimum_value, to = maximum_value, by = grid_interval)
for (b in 1:length(data_interval)) {
  x_left <- data_interval[b]
  x_right <- data_interval[b+1]
  lines(c(x_left, x_right), c(get_prediction(x_left, is_terminal, node_splits, node_means), get_prediction(x_left, is_terminal, node_splits, node_means)), lwd = 2, col = "black")
  if (b < length(data_interval)) {
    lines(c(x_right, x_right), c(get_prediction(x_left, is_terminal, node_splits, node_means), get_prediction(x_right, is_terminal, node_splits, node_means)), lwd = 2, col = "black") 
  }
}

# QUESTION 4- Calculate RMSE for test data points
y_test_predicted <- sapply(X=1:N_test, FUN = function(i) get_prediction(x_test[i], is_terminal, node_splits, node_means))
RMSE <- sqrt(sum((y_test - y_test_predicted) ^ 2) / length(y_test))
sprintf("RMSE is %s when P is %s", RMSE, P)

# QUESTION 5 - P vs RMSE
RMSEs <- sapply(X=1:20, FUN = function(p) {
  sprintf("Calculating RMSE for %d", p)
  result <- DecisionTreeRegression(p)
  node_splits <- result$splits
  node_means <- result$means
  is_terminal <- result$is_terminal
  y_test_predicted <- sapply(X=1:N_test, FUN = function(i) get_prediction(x_test[i], is_terminal, node_splits, node_means))
  RMSE <- sqrt(sum((y_test - y_test_predicted) ^ 2) / length(y_test))
})

plot(1:20, RMSEs,
     type = "o", lwd = 1, las = 1, pch = 1, lty = 2,
     xlab = "P", ylab = "RMSE")
