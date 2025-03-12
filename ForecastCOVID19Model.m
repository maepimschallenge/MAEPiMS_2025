%% COVID-19 Model Forecast for Nigeria 2022
% This script loads the fitted model parameters and forecasts COVID-19 cases
% for Nigeria in 2022 based on the fitted 2020-2021 data

clear;
close all;

%% Load the fitted model parameters
if ~exist('COVID19_fit_params.mat', 'file')
    error('Fitted parameters file not found. Run FitCOVID19Model.m first.');
end

load('COVID19_fit_params.mat');
disp('Loaded fitted parameters from COVID19_fit_params.mat');

%% Define the forecast period
% Original data period (Jul 9 - Aug 7, 2021)
% Fixed datetime format with 'InputFormat' parameter
start_date = datetime('09-Jul-2021', 'InputFormat', 'dd-MMM-yyyy');
end_date = datetime('07-Aug-2021', 'InputFormat', 'dd-MMM-yyyy');

% Forecast from end of data to end of 2022
forecast_start = end_date + days(1);
forecast_end = datetime('31-Dec-2022', 'InputFormat', 'dd-MMM-yyyy');

% Total simulation period (original data + forecast)
full_start = start_date;
full_end = forecast_end;

% Convert to days for simulation
days_original = 0:days(end_date - start_date);
days_forecast = days(forecast_start - start_date):days(forecast_end - start_date);
days_full = 0:days(forecast_end - start_date);

%% Simulate with the fitted parameters for the full period
tspan = [0 days(forecast_end - start_date)];

% Use ode45 to simulate the model
[t, y] = ode45(@(t, y) COVID19Model(t, y, params), tspan, initial_conditions);

%% Extract and process the results
% Infected cases over time
I_full = y(:, 3);

% Original data period
I_original = I_full(t <= max(days_original));
t_original = t(t <= max(days_original));

% Forecast period
I_forecast = I_full(t >= min(days_forecast));
t_forecast = t(t >= min(days_forecast));

%% Plot the results
figure('Position', [100, 100, 1200, 800]);

% Original data with actual values
subplot(2, 1, 1);
dates_original = start_date + days(days_original);
% Actual COVID-19 cases from the data
cases = [11713, 11515, 11421, 10357, 10363, 10243, 10237, 10126, 9174, 9231, ...
         9170, 9202, 9139, 9227, 7700, 7594, 7518, 7520, 7578, 7626, ...
         7599, 7625, 7669, 7734, 7782, 7821, 7840, 7986, 7949, 7929];

scatter(dates_original, cases, 100, 'b', 'filled');
hold on;
plot(start_date + days(t_original), I_original, 'r-', 'LineWidth', 4);
title('COVID-19 Model: Fit to 2021 Data', 'FontSize', 23);
xlabel('Date', 'FontSize', 23);
ylabel('Number of Infected Cases', 'FontSize', 23);
legend('Actual Data', 'Model Fit', 'FontSize', 16, 'Location', 'best');
grid on;
datetick('x', 'mmm-yy');
xlim([start_date, end_date]);
ax = gca;
ax.FontSize = 16;
ax.LineWidth = 1.5;

% Forecast for 2022
subplot(2, 1, 2);
dates_forecast = start_date + days(t_forecast);

% Plot forecast with confidence intervals
plot(dates_forecast, I_forecast, 'g-', 'LineWidth', 4);
hold on;

% Add uncertainty bands (assuming 15% uncertainty)
I_upper = I_forecast * 1.15;
I_lower = I_forecast * 0.85;
fill([dates_forecast; flipud(dates_forecast)], [I_upper; flipud(I_lower)], 'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

title('COVID-19 Model: Forecast for 2022', 'FontSize', 23);
xlabel('Date', 'FontSize', 23);
ylabel('Predicted Number of Infected Cases', 'FontSize', 23);
legend('Forecast', '15% Uncertainty Range', 'FontSize', 16, 'Location', 'best');
grid on;
datetick('x', 'mmm-yy');
xlim([forecast_start, forecast_end]);
ax = gca;
ax.FontSize = 16;
ax.LineWidth = 1.5;

%% Calculate forecast statistics
forecast_mean = mean(I_forecast);
forecast_max = max(I_forecast);
forecast_min = min(I_forecast);
forecast_std = std(I_forecast);

% Display forecast statistics
disp('Forecast Statistics for 2022:');
disp(['Mean Infected Cases: ', num2str(forecast_mean)]);
disp(['Maximum Infected Cases: ', num2str(forecast_max)]);
disp(['Minimum Infected Cases: ', num2str(forecast_min)]);
disp(['Standard Deviation: ', num2str(forecast_std)]);

%% Save forecast results
forecast_results = struct();
forecast_results.dates = dates_forecast;
forecast_results.infected = I_forecast;
forecast_results.upper_bound = I_upper;
forecast_results.lower_bound = I_lower;
forecast_results.statistics.mean = forecast_mean;
forecast_results.statistics.max = forecast_max;
forecast_results.statistics.min = forecast_min;
forecast_results.statistics.std = forecast_std;

save('COVID19_forecast_results.mat', 'forecast_results');
disp('Forecast results saved to COVID19_forecast_results.mat');

%% Create a monthly summary table for 2022
% Generate monthly dates for 2022
monthly_dates = datetime(2022, 1:12, 1);

% Initialize arrays for monthly statistics
monthly_mean = zeros(length(monthly_dates), 1);
monthly_max = zeros(length(monthly_dates), 1);
monthly_min = zeros(length(monthly_dates), 1);

% Calculate monthly statistics
for i = 1:length(monthly_dates)
    month_start = datetime(2022, i, 1);
    if i < 12
        month_end = datetime(2022, i+1, 1) - days(1);
    else
        month_end = datetime(2022, 12, 31);
    end
    
    % Find indices within this month
    idx = (dates_forecast >= month_start) & (dates_forecast <= month_end);
    
    if sum(idx) > 0
        monthly_mean(i) = mean(I_forecast(idx));
        monthly_max(i) = max(I_forecast(idx));
        monthly_min(i) = min(I_forecast(idx));
    end
end

% Create a table with monthly statistics
month_names = {'January', 'February', 'March', 'April', 'May', 'June', ...
               'July', 'August', 'September', 'October', 'November', 'December'};
monthly_table = table(month_names', monthly_mean, monthly_max, monthly_min, ...
                     'VariableNames', {'Month', 'Mean_Cases', 'Max_Cases', 'Min_Cases'});

% Display the table
disp('Monthly COVID-19 Case Forecast for 2022:');
disp(monthly_table);

% Save the table to a file
writetable(monthly_table, 'COVID19_monthly_forecast_2022.csv');
disp('Monthly forecast saved to COVID19_monthly_forecast_2022.csv');

%% COVID19Model Function
function dydt = COVID19Model(t, y, params)
% COVID19Model ODE function for simulating a compartmental COVID-19 model
%
% Inputs:
%   t - Current time point
%   y - Current state vector [S, E, I, Q, J, T, V, R]
%   params - Structure containing model parameters
%
% Output:
%   dydt - Derivatives of state variables

% Extract state variables
S = y(1); % Susceptible
E = y(2); % Exposed
I = y(3); % Infected
Q = y(4); % Quarantined
J = y(5); % Hospitalized
T = y(6); % Treated
V = y(7); % Vaccinated
R = y(8); % Recovered

% Total population
N = sum(y);

% Extract parameters from structure
Lambda = params.Lambda;       % Recruitment rate
beta_1 = params.beta_1;       % Infection rate from infected
beta_2 = params.beta_2;       % Infection rate from hospitalized
chi_1 = params.chi_1;         % Vaccination rate
chi_2 = params.chi_2;         % Waning immunity rate from vaccination
psi_1 = params.psi_1;         % Rate at which quarantined move to hospitalization
psi_2 = params.psi_2;         % Rate at which quarantined return to susceptible
eta = params.eta;             % Rate of loss of immunity
mu = params.mu;               % Natural death rate
mu_prime = params.mu_prime;   % Death rate for treated class
lambda = params.lambda;       % Rate at which exposed become infected
phi = params.phi;             % Rate at which exposed are quarantined
sigma_1 = params.sigma_1;     % Rate at which infected are hospitalized
sigma_2 = params.sigma_2;     % Rate at which infected are treated
delta_1 = params.delta_1;     % Disease-induced death rate for infected
delta_2 = params.delta_2;     % Disease-induced death rate for treated
delta_3 = params.delta_3;     % Disease-induced death rate for hospitalized
gamma = params.gamma;         % Rate at which hospitalized move to treatment
omega = params.omega;         % Recovery rate of treated individuals

% Calculate a_1 (force of infection)
a_1 = (beta_1 * I + beta_2 * J) / N;

% System of differential equations
dydt = zeros(8, 1);

% dS/dt: Change in susceptible population
dydt(1) = Lambda - a_1 * S - chi_1 * S + chi_2 * V + psi_2 * Q + eta * R - mu * S;

% dE/dt: Change in exposed population
dydt(2) = a_1 * S - (lambda + phi + mu) * E;

% dI/dt: Change in infected population
dydt(3) = lambda * E - (sigma_1 + sigma_2 + delta_1 + mu) * I;

% dQ/dt: Change in quarantined population
dydt(4) = phi * E - (psi_1 + psi_2 + mu) * Q;

% dJ/dt: Change in hospitalized population
dydt(5) = psi_1 * Q + sigma_1 * I - (gamma + delta_3 + mu) * J;

% dT/dt: Change in treated population
dydt(6) = sigma_2 * I + gamma * J - (omega + delta_2 + mu_prime) * T;

% dV/dt: Change in vaccinated population
dydt(7) = chi_1 * S - (chi_2 + mu) * V;

% dR/dt: Change in recovered population
dydt(8) = omega * T - (eta + mu) * R;
end
