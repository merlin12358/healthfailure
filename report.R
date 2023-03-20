library(tidyverse)
library(lubridate)
library(ggplot2)
library(devtools)
library(skimr)
library(GGally)
library(caret)
library(randomForest)
library(data.table)
library(mlr3verse)
library(tidyverse)
library(corrplot)
library(rms)
# Load data
data <- read.csv("https://www.louisaslett.com/Courses/MISCADA/heart_failure.csv")
# Summarize data
summary(data)
skim(data)
library(recipes)
cake <- recipe(fatal_mi ~ ., data = data) %>%
  step_impute_mean(all_numeric()) %>% # impute missings on numeric values with the mean
  step_center(all_numeric()) %>% # center by subtracting the mean from all numeric features
  step_scale(all_numeric()) %>% # scale by dividing by the standard deviation on all numeric features
  step_unknown(all_nominal(), -all_outcomes()) %>% # create a new factor level called "unknown" to account for NAs in factors, except for the outcome (response can't be NA)
  step_dummy(all_nominal(), one_hot = TRUE) %>% # turn all factors into a one-hot coding
  prep(training = data_train) # learn all the parameters of preprocessing on the training data

data_train_final <- bake(cake, new_data = data_train) # apply preprocessing to training data
data_validate_final <- bake(cake, new_data = data_validate) # apply preprocessing to validation data
data_test_final <- bake(cake, new_data = data_test) # apply preprocessing to testing data
view(as.data.table(data_train_final))
data$fatal_mi <- as.factor(data$fatal_mi)
# Logistic Regression
set.seed(212) # set seed for reproducibility
data_task <- TaskClassif$new(id = "HeartFailure",
                             backend = data, # <- NB: no na.omit() this time
                             target = "fatal_mi",
                             positive = "1")
cv5 <- rsmp("cv", folds = 5)

# Set up cross-validation control
cv_control <- trainControl(method = "cv", number = 5)
data_validate$fatal_mi <- as.factor(data_validate$fatal_mi)
data_test$fatal_mi <- as.factor(data_test$fatal_mi)
# Normalize continuous features using the 'preProcess' function from the 'caret' package
continuous_features <- c("age", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium")
preprocessing_params <- preProcess(data[, continuous_features], method = c("center", "scale"))

# Apply the normalization
data_normalized <- predict(preprocessing_params, data[, continuous_features])

# Combine the normalized continuous features with the categorical features and target variable
data_normalized <- cbind(data_normalized, data[, c("anaemia", "diabetes", "high_blood_pressure", "sex", "smoking", "time", "fatal_mi")])
# Convert the 'fatal_mi' column to a factor
data_normalized$fatal_mi <- as.factor(data_normalized$fatal_mi)

# Modify factor levels to valid R variable names
levels(data_normalized$fatal_mi) <- c("No_Fatal_MI", "Fatal_MI")

# Create an 80-20 train-test split
set.seed(123)
split <- createDataPartition(data_normalized$fatal_mi, p = 0.8, list = FALSE)
train_data <- data_normalized[split, ]
test_data <- data_normalized[-split, ]

# Define the training control using 5-fold cross-validation
train_control <- trainControl(method = "cv", number = 5, savePredictions = "final", classProbs = TRUE)

# Define the evaluation metric (Accuracy)
metric <- "Accuracy"
# Logistic Regression
set.seed(123)
logistic_regression <- train(
  fatal_mi ~ ., data = train_data,
  method = "glm",
  trControl = train_control,
  metric = metric
)
# Print the model performance
print(logistic_regression)

# Random Forest
set.seed(123)
random_forest <- train(
  fatal_mi ~ ., data = train_data,
  method = "rf",
  trControl = train_control,
  metric = metric
)
print(random_forest)

# Neural Networks
set.seed(123)
nnet_model <- train(
  fatal_mi ~ ., data = train_data,
  method = "nnet",
  trControl = train_control,
  metric = metric
)
print(nnet_model)
test_predictions <- predict(logistic_regression, newdata = test_data)
ggplot(data.frame(x = test_predictions), aes(stat="count"),) + geom_histogram()
confusion_matrix <- confusionMatrix(test_predictions, test_data$fatal_mi)
# Print the confusion matrix and test set performance metrics
print(confusion_matrix)
logreg_grid <- expand.grid(
  .alpha = seq(0, 1, length.out = 11), # Regularization type: 0 for Ridge and 1 for Lasso
  .lambda = seq(0.01, 1, length.out = 10) # Regularization strength
)
logreg_tuned <- train(
  fatal_mi ~ ., data = data_normalized,
  method = "glmnet",
  trControl = train_control,
  metric = "Accuracy",
  tuneGrid = logreg_grid
)
# Display tuning results
print(logreg_tuned)
plot(logreg_tuned)
logreg_rms <- lrm(fatal_mi ~ ., data = data_normalized, x = TRUE,y = TRUE)
# Prepare the dataset for the validation
# Fit the logistic regression model using the rms package
logreg_rms <- lrm(fatal_mi ~ ., data = data_normalized, x = TRUE, y = TRUE)

# Set up the validation function
val <- validate(logreg_rms, B = 200, method = "boot", dxy = TRUE, pr = TRUE)

# Create a calibration plot using the original logreg_rms model
cal <- calibrate(logreg_rms, B = 200, cmethod = "hare")
plot(cal, xlab = "Predicted Probability", ylab = "Observed Probability", main = "Calibration Plot for Logistic Regression")
