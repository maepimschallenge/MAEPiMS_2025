library(tidyverse)
library(readxl)
library(MMWRweek)
library(forecast)
library(prophet)
library(xgboost)
library(Metrics)
library(reshape2)
library(zoo)  # For rolling averages


# Load dataset
mydata <- read_excel("C:/Users/Agba Xchanger/Documents/Datasets/covid-19_dataset.xlsx")

# Remove unnecessary columns
data_main <- mydata %>%
  select(-c(extreme_poverty, hospital_beds_per_thousand, population))

# Extract data for 2020-2021 (727 days)
data <- head(data_main, 727)

# Remove additional unwanted columns
data <- data %>%
  select(-c(population_density, median_age, aged_65_older, aged_70_older, gdp_per_capita, 
            cardiovasc_death_rate, diabetes_prevalence, female_smokers, male_smokers,
            handwashing_facilities, life_expectancy, human_development_index))

# Add epidemiological week and year columns
data$epi_year <- MMWRweek(data$date)$MMWRyear
data$epi_week <- MMWRweek(data$date)$MMWRweek


# Aggregate data by epidemiological week
weekly_data <- data %>%
  group_by(epi_year, epi_week) %>%
  summarise(
    total_cases = sum(new_cases, na.rm = TRUE),
    total_deaths = sum(new_deaths, na.rm = TRUE),
    avg_stringency = mean(stringency_index, na.rm = TRUE)
  )

# Plot Weekly Cases Over Time
ggplot(weekly_data, aes(x = epi_week, y = total_cases, group = epi_year, color = as.factor(epi_year))) +
  geom_line() +
  labs(title = "Weekly COVID-19 Cases in Nigeria", x = "Epidemiological Week", y = "Total Cases") +
  theme_minimal()


# Convert to time series
# Convert to time series (weekly seasonality)
ts_data <- ts(data$new_cases_smoothed, start = c(2020, 1), frequency = 52)

# Check for missing values & fix them
ts_data[is.na(ts_data)] <- 0
ts_data[!is.finite(ts_data)] <- mean(ts_data, na.rm = TRUE)

# Plot time series to check trend
autoplot(ts_data) + labs(title = "COVID-19 Cases Time Series")

# Check for missing or infinite values
sum(is.na(ts_data))  # Count NA values
sum(!is.finite(ts_data))  # Count Inf/-Inf values

# Convert Time Series (Try Different Seasonal Periods)
ts_data <- ts(data$new_cases_smoothed, start = c(2020, 1), frequency = 7)  # Weekly
# ts_data <- ts(data$new_cases_smoothed, start = c(2020, 1), frequency = 365)  # Monthly (Test this too)

# Fit SARIMA Model (Auto-Detect Best Order)
sarima_model <- auto.arima(ts_data, seasonal = TRUE, stepwise = FALSE, approximation = FALSE)

# Print model summary
summary(sarima_model)

# Generate SARIMA Forecast for 2022 (365 days ahead)
forecast_sarima <- forecast(sarima_model, h = 365)

# Convert forecast to a DataFrame
forecast_sarima <- data.frame(
  date = seq(max(data$date) + 1, by = "day", length.out = 365),
  predicted_cases = as.numeric(forecast_sarima$mean)
)

# Print first 10 rows to verify
head(forecast_sarima, 10)


# Prepare Prophet dataset
prophet_data <- data.frame(ds = data$date, y = data$new_cases)

# Fit Prophet model with yearly & weekly seasonality
prophet_model <- prophet(yearly.seasonality = TRUE, weekly.seasonality = TRUE, daily.seasonality = TRUE)
prophet_model <- fit.prophet(prophet_model, prophet_data)

# Summary of the model
summary(prophet_model)

# Predict for 2022
future <- make_future_dataframe(prophet_model, periods = 365)
forecast_prophet <- predict(prophet_model, future)

# Extract trend & seasonality
trend_data <- forecast_prophet[, c("ds", "trend")]
seasonality_data <- forecast_prophet[, c("ds", "trend", "additive_terms")]

# Plot Prophet forecast
plot(prophet_model, forecast_prophet) +
  labs(title = "Prophet Forecast for COVID-19 Cases in Nigeria",
       x = "Date", y = "Predicted Cases")


### XGBOOST MODEL

# Add New Features to Enhance Variability
data <- data %>%
  mutate(
    lag_1 = lag(new_cases_smoothed, 1),
    lag_2 = lag(new_cases_smoothed, 2),
    lag_3 = lag(new_cases_smoothed, 3),
    lag_7 = lag(new_cases_smoothed, 7),
    day_of_week = as.numeric(format(date, "%u")),
    month = as.numeric(format(date, "%m")),
    rolling_avg_7 = zoo::rollmean(new_cases_smoothed, k = 7, fill = NA, align = "right"),
    rolling_avg_14 = zoo::rollmean(new_cases_smoothed, k = 14, fill = NA, align = "right"),
    diff_cases = new_cases_smoothed - lag(new_cases_smoothed, 1)  # First difference
  ) %>%
  na.omit()  # Remove NA values from moving averages and lags


# Define input (X) and output (y)
X <- as.matrix(data[, c("lag_1", "lag_2", "lag_3", "lag_7", "stringency_index", "day_of_week", "month", "rolling_avg_7")])
y <- as.numeric(data$new_cases_smoothed)


# Train-Test Split
train_size <- round(0.8 * length(y))  # Use `length(y)` instead of `nrow(y)`

X_train <- X[1:train_size, ]
y_train <- y[1:train_size]

X_test <- X[(train_size+1):length(y), ]  # Use `length(y)`
y_test <- y[(train_size+1):length(y)]  # Use `length(y)`

# Check if the split sizes are correct
print(dim(X_train))
print(length(y_train))
print(dim(X_test))
print(length(y_test))


# Train XGBoost Model
xgb_model <- xgboost(data = X_train, label = y_train, nrounds = 200, max_depth = 6, eta = 0.05, objective = "reg:squarederror")

# Predict on Test Set
predictions_xgb <- predict(xgb_model, X_test)

# Rolling Forecast for 2022
final_predictions <- numeric(365)
X_future <- X_test[nrow(X_test), , drop = FALSE]  # Start with last test row

for (i in 1:365) {
  X_future_dmatrix <- xgb.DMatrix(data = as.matrix(X_future))  
  final_predictions[i] <- predict(xgb_model, X_future_dmatrix)
  
  # Shift values in X_future dynamically
  new_row <- c(
    final_predictions[i], X_future[1, 1], X_future[1, 2],  # Update lag_1, lag_2
    X_future[1, 3], X_future[1, 4],                        # Shift lag_3, lag_7
    X_future[1, 5:7],                                      # Keep time-based features
    mean(final_predictions[max(1, i - 6):i])  # Compute rolling 7-day average dynamically
  )
  
  X_future <- rbind(X_future, new_row)[-1, , drop = FALSE]
}

unique(final_predictions)  # Should contain different values


# # Save Predictions
# final_results <- data.frame(
#   date = seq(max(data$date) + 1, by = "day", length.out = 365),
#   predicted_cases = final_predictions
# )
# write.csv(final_results, "final_predictions_xgboost.csv", row.names = FALSE)


subset_data <- data_main[728:1092, ]

subset_data <- subset(subset_data, select = -c(population_density, median_age, aged_65_older, aged_70_older, gdp_per_capita, 
                                               cardiovasc_death_rate, diabetes_prevalence, female_smokers, male_smokers,
                                               handwashing_facilities, life_expectancy, human_development_index))

# Add epidemiological week and year columns
subset_data$epi_year <- MMWRweek(subset_data$date)$MMWRyear
subset_data$epi_week <- MMWRweek(subset_data$date)$MMWRweek

# Ensure all variables have exactly 365 rows
print(length(subset_data$new_cases_smoothed))  # Check actual values
print(length(forecast_sarima$predicted_cases))  # Check SARIMA predictions
print(length(forecast_prophet$yhat[(nrow(data)+1):(nrow(data)+365)]))  # Check Prophet predictions
print(length(final_predictions))  # Check XGBoost predictions

# Fix Prophet Predictions if needed
prophet_pred <- forecast_prophet$yhat[(nrow(data)+1):min(nrow(forecast_prophet), nrow(data) + 365)]
if (length(prophet_pred) < 365) {
  prophet_pred <- c(prophet_pred, rep(NA, 365 - length(prophet_pred)))  # Fill missing values
}

# Create Comparison DataFrame
comparison_df <- data.frame(
  date = seq(as.Date("2022-01-01"), by = "day", length.out = 365),
  
  actual = subset_data$new_cases_smoothed,  # Use actual 2022 values from subset_data
  
  sarima_predicted = forecast_sarima$predicted_cases,  # SARIMA Predictions
  
  prophet_predicted = prophet_pred,  # Prophet Predictions
  
  xgb_predicted = final_predictions  # XGBoost Predictions
)

# Verify the structure
print(dim(comparison_df))  # Should be (365, 5)


# Print First 10 Rows
head(comparison_df, 10)

## Saving the forecast in a CSV file
write.csv(comparison_df, "final_models_predictions.csv", row.names = FALSE)


### Comparing the models

actual_values <- subset_data$new_cases_smoothed

# Calculate Errors for Each Model
mae_sarima <- mae(actual_values, forecast_sarima$predicted_cases)
rmse_sarima <- rmse(actual_values, forecast_sarima$predicted_cases)

mae_prophet <- mae(actual_values, prophet_pred)
rmse_prophet <- rmse(actual_values, prophet_pred)

mae_xgb <- mae(actual_values, final_predictions)
rmse_xgb <- rmse(actual_values, final_predictions)

# Summary Table
model_comparison <- data.frame(
  Model = c("SARIMA", "Prophet", "XGBoost"),
  MAE = c(mae_sarima, mae_prophet, mae_xgb),
  RMSE = c(rmse_sarima, rmse_prophet, rmse_xgb)
)
print(model_comparison)


# Convert model comparison table for plotting
model_comparison_melted <- melt(model_comparison, id = "Model")

# Create a bar chart for MAE & RMSE
ggplot(model_comparison_melted, aes(x = Model, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Model Performance Comparison",
       x = "Model",
       y = "Error Metric Value",
       fill = "Metric") +
  theme_minimal()


# Plot Actual vs. Predicted Cases
ggplot(comparison_df, aes(x = date)) +
  
  # Actual Cases in Black
  geom_line(aes(y = actual, color = "Actual Cases"), size = 1) +
  
  # Prophet Predictions in Blue
  geom_line(aes(y = prophet_predicted, color = "Prophet Predictions"), linetype = "dashed", size = 1) +
  
  # XGBoost Predictions in Red
  geom_line(aes(y = xgb_predicted, color = "XGBoost Predictions"), linetype = "dotdash", size = 1) +
  
  labs(title = "Prophet vs XGBoost: COVID-19 Predictions (2022)",
       x = "Date", y = "Cases", color = "Legend") +
  
  scale_color_manual(values = c(
    "Actual Cases" = "black",
    "Prophet Predictions" = "blue",
    "XGBoost Predictions" = "red"
  )) +
  
  theme_minimal()

