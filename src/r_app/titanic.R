suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

# Locate src/data relative to this script, so it works from any folder and in Docker
args <- commandArgs(trailingOnly = FALSE)
script_arg <- sub("^--file=", "", args[grep("^--file=", args)])
script_dir <- if (length(script_arg) == 1) dirname(normalizePath(script_arg)) else getwd()
data_dir <- normalizePath(file.path(script_dir, "..", "data"), mustWork = FALSE)

cat(strrep("=", 60), "\n", sep = "")
cat("Problem 23: Load packages and train.csv\n")
cat(strrep("=", 60), "\n", sep = "")
cat("R version:", R.version.string, "\n")
cat("readr version:", as.character(packageVersion("readr")),
    "| dplyr version:", as.character(packageVersion("dplyr")), "\n")
cat("Reading data from:", data_dir, "\n")

train <- read_csv(file.path(data_dir, "train.csv"), show_col_types = FALSE)
cat(sprintf("train.csv loaded: %d rows, %d columns\n", nrow(train), ncol(train)))
cat("\nFirst 5 rows:\n")
print(head(train, 5), width = 120)

cat("\n", strrep("=", 60), "\n", sep = "")
cat("Problem 14 (R): Explore, add, and adjust the data\n")
cat(strrep("=", 60), "\n", sep = "")

# --- 14a. Explore all three files ---
for (name in c("train.csv", "test.csv", "gender_submission.csv")) {
  df <- read_csv(file.path(data_dir, name), show_col_types = FALSE)
  cat(sprintf("\n===== %s: %d rows, %d columns =====\n", name, nrow(df), ncol(df)))
  profile <- data.frame(
    type = sapply(df, function(x) class(x)[1]),
    missing = colSums(is.na(df)),
    n_unique = sapply(df, function(x) length(unique(na.omit(x))))
  )
  print(profile)
}

cat(sprintf("\n[Explore] Overall survival rate in train: %.1f%%\n", 100 * mean(train$Survived)))
cat("\n[Explore] Survival rate by sex:\n")
print(as.data.frame(train %>% group_by(Sex) %>% summarise(survival_rate = round(mean(Survived), 3))))
cat("\n[Explore] Survival rate by passenger class:\n")
print(as.data.frame(train %>% group_by(Pclass) %>% summarise(survival_rate = round(mean(Survived), 3))))

# --- 14b-d. Add features, fill missing values, encode (reused for test in Problem 17) ---
prepare <- function(df, label, fill = NULL) {
  cat(sprintf("\n----- Preparing %s data: %d rows -----\n", label, nrow(df)))

  # Add features
  df <- df %>%
    mutate(
      Title = trimws(sub(".*,\\s*([^.]+)\\..*", "\\1", Name)),
      Title = recode(Title, Mlle = "Miss", Ms = "Miss", Mme = "Mrs"),
      Title = if_else(Title %in% c("Mr", "Mrs", "Miss", "Master"), Title, "Rare"),
      FamilySize = SibSp + Parch + 1,
      IsAlone = as.integer(FamilySize == 1),
      HasCabin = as.integer(!is.na(Cabin))
    )
  cat(sprintf("\n[%s] Title extracted from Name (rare titles grouped as 'Rare'):\n", label))
  print(table(df$Title))
  cat(sprintf("\n[%s] FamilySize = SibSp + Parch + 1\n", label))
  print(summary(df$FamilySize))
  cat(sprintf("[%s] IsAlone: %d\n", label, sum(df$IsAlone)))
  cat(sprintf("[%s] HasCabin: %d of %d passengers\n", label, sum(df$HasCabin), nrow(df)))

  # Learn fill values from train only
  if (is.null(fill)) {
    fill <- list(
      age_by_title = df %>% group_by(Title) %>% summarise(age_fill = median(Age, na.rm = TRUE)),
      embarked = names(which.max(table(df$Embarked))),
      fare = median(df$Fare, na.rm = TRUE)
    )
  }

  # Fill missing values
  n_age <- sum(is.na(df$Age)); n_emb <- sum(is.na(df$Embarked)); n_fare <- sum(is.na(df$Fare))
  df <- df %>%
    left_join(fill$age_by_title, by = "Title") %>%
    mutate(
      Age = coalesce(Age, age_fill),
      Embarked = coalesce(Embarked, fill$embarked),
      Fare = coalesce(Fare, fill$fare)
    ) %>%
    select(-age_fill)
  cat(sprintf("\n[%s] Filled %d missing Age values with the median age of each Title (from train):\n", label, n_age))
  print(as.data.frame(fill$age_by_title))
  cat(sprintf("\n[%s] Filled %d missing Embarked values with the most common port: %s\n", label, n_emb, fill$embarked))
  cat(sprintf("[%s] Filled %d missing Fare values with the median fare from train: %.2f\n", label, n_fare, fill$fare))

  # Encode and drop
  df <- df %>%
    mutate(Sex = as.integer(Sex == "female")) %>%
    select(-Name, -Ticket, -Cabin)
  cat(sprintf("\n[%s] Encoded Sex: male = 0, female = 1\n", label))
  cat(sprintf("[%s] Dropped columns: Name, Ticket, Cabin\n", label))
  cat(sprintf("[%s] Remaining missing values: %d\n", label, sum(is.na(df))))

  list(df = df, fill = fill)
}

prep <- prepare(train, "Train")
train <- prep$df
fill <- prep$fill
cat("\n")
print(head(as.data.frame(train), 5))

cat("\n", strrep("=", 60), "\n", sep = "")
cat("Problem 15 (R): Build a logistic regression model\n")
cat(strrep("=", 60), "\n", sep = "")

# Turn the text columns into factors with fixed levels, so train and test always match.
# The first level is the baseline (Master for Title, C for Embarked), like drop_first in Python.
to_model_data <- function(df) {
  df %>% mutate(
    Title = factor(Title, levels = c("Master", "Miss", "Mr", "Mrs", "Rare")),
    Embarked = factor(Embarked, levels = c("C", "Q", "S"))
  )
}

train_model <- to_model_data(train)

# Same features as the Python model: Sex is left out because Title already encodes it,
# and SibSp/Parch are left out because FamilySize combines them.
model_formula <- Survived ~ Pclass + Age + Fare + FamilySize + IsAlone + HasCabin + Title + Embarked
cat("[Model] Formula:", deparse(model_formula), "\n")
cat(sprintf("[Model] Training rows: %d, target: Survived (1 = survived)\n", nrow(train_model)))

model <- glm(model_formula, data = train_model, family = binomial)
cat("[Model] Fitted logistic regression with glm(family = binomial)\n\n")
print(summary(model))

cat("\n[Model] Odds ratios (exp of coefficients):\n")
print(round(exp(coef(model)), 3))

# ------------------------------------------------------------
cat("\n", strrep("=", 60), "\n", sep = "")
cat("Problem 16 (R): Accuracy on the training set\n")
cat(strrep("=", 60), "\n", sep = "")

train_prob <- predict(model, train_model, type = "response")
train_pred <- as.integer(train_prob > 0.5)
train_acc <- mean(train_pred == train_model$Survived)
cat(sprintf("[Train] Accuracy: %.4f (%d of %d correct)\n",
            train_acc, sum(train_pred == train_model$Survived), nrow(train_model)))

cat("\n[Train] Confusion matrix (rows = actual, columns = predicted):\n")
print(table(Actual = train_model$Survived, Predicted = train_pred))

majority_acc <- mean(train_model$Survived == 0)
gender_acc <- mean(train_model$Sex == train_model$Survived)
cat(sprintf("\n[Train] Baseline - predict everyone died:   %.4f\n", majority_acc))
cat(sprintf("[Train] Baseline - women survive, men die:  %.4f\n", gender_acc))

# ------------------------------------------------------------
cat("\n", strrep("=", 60), "\n", sep = "")
cat("Problem 17 (R): Load test.csv and predict\n")
cat(strrep("=", 60), "\n", sep = "")

test <- read_csv(file.path(data_dir, "test.csv"), show_col_types = FALSE)
cat(sprintf("[Test] test.csv loaded: %d rows, %d columns\n", nrow(test), ncol(test)))

# Same steps as Problem 14, using the fill values learned from train
test <- prepare(test, "Test", fill)$df
test_model <- to_model_data(test)

test_prob <- predict(model, test_model, type = "response")
test_pred <- as.integer(test_prob > 0.5)
predictions <- data.frame(PassengerId = test_model$PassengerId, Survived = test_pred,
                          SurvivalProb = round(test_prob, 3))
cat(sprintf("\n[Test] Predicted %d passengers: %d survived (%.1f%%), %d died\n",
            nrow(predictions), sum(test_pred), 100 * mean(test_pred), sum(test_pred == 0)))
print(head(predictions, 5))

# --- 18 ---
cat("\n", strrep("=", 60), "\n", sep = "")
cat("Problem 18 (R): Predictions on the test set\n")
cat(strrep("=", 60), "\n", sep = "")
cat("[Test] test.csv has no Survived labels, so test accuracy cannot be measured.\n")
cat("       Printing the predictions for all test passengers instead.\n\n")
print(predictions, row.names = FALSE)
cat(sprintf("\n[Test] Total: %d passengers, %d predicted to survive, %d predicted to die\n",
            nrow(predictions), sum(predictions$Survived), sum(predictions$Survived == 0)))
            