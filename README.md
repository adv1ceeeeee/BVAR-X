# BVAR-X: Bayesian Vector Autoregression with Exogenous Variables

## Overview
`BVAR-X` is a Python implementation of a Bayesian Vector Autoregression (BVAR) model with support for exogenous variables. This package provides tools for estimating BVAR models, forecasting, hyperparameter optimization, and visualization of results. It supports multiple prior distributions, including Litterman-Minnesota, Normal-Flat, Normal-Wishart, and Sims-Zha priors (both Normal-Flat and Normal-Wishart). The code is designed to handle real-world financial datasets, such as those stored in Excel, and includes a user-friendly console interface for interactive model configuration.

## Features
- **Multiple Prior Distributions**: Supports Litterman-Minnesota, Normal-Flat, Normal-Wishart, Sims-Zha Normal-Flat, and Sims-Zha Normal-Wishart priors.
- **Exogenous Variables**: Incorporates exogenous variables in both estimation and forecasting.
- **Hyperparameter Optimization**: Automatically optimizes hyperparameters for each prior using cross-validation and mean squared error.
- **Forecasting**: Generates median forecasts with confidence intervals based on Monte Carlo simulations.
- **Data Preprocessing**: Handles data normalization (log-transformation and standardization) with inverse transformation for results.
- **Visualization**: Plots historical data, forecasts, and confidence intervals with customizable variable selection.
- **Model Selection**: Computes BIC and AIC for lag order selection and supports modal lag selection across multiple priors.
- **Excel Data Integration**: Loads and preprocesses data from Excel files with automatic cleaning and handling of missing values.
- **Interactive Interface**: Console-based user interface for configuring the model, selecting priors, and specifying forecast horizons.

## Requirements
- Python 3.7+
- Required libraries:
  - `numpy`
  - `pandas`
  - `scipy`
  - `matplotlib`
  - `sklearn`
  - `statsmodels` (optional, for ARIMA-based exogenous variable forecasting)
- Excel file support requires `openpyxl` or `xlrd`.

Install dependencies using:
```bash
pip install numpy pandas scipy matplotlib scikit-learn statsmodels openpyxl
```

## Usage
The main entry point is the `user_interface()` function, which provides a console-based interface for configuring and running the BVAR model. The code expects an Excel file with financial data (e.g., portfolio metrics) in a specific format.

### Steps to Run
1. **Prepare Your Data**:
   - Ensure your Excel file has a sheet (e.g., named "Eviews") with columns: `Month`, `Uninvestedfunds`, `NettoFunds`, `Reinvestedfunds`, `Totalclients`, `Investedfunds`, and `Plannedrate`.
   - The `Month` column should be in `DD.MM.YYYY` format.
   - Example file path: `"C:/Users/YourPath/Портфель.xlsx"`.

2. **Run the Script**:
   - Update the `file_path` in the `user_interface()` function to point to your Excel file.
   - Execute the script:
     ```bash
     python BVAR-X.py
     ```

3. **Follow the Console Prompts**:
   - Load data from the specified Excel file.
   - Choose the lag order (`p`) manually or optimize using BIC/AIC.
   - Select a prior distribution or use automatic selection/median forecasting.
   - Configure hyperparameters (default, manual, or optimized).
   - Specify variables for visualization (endogenous and exogenous).
   - Generate and view forecasts with confidence intervals.

### Example Workflow
```python
if __name__ == "__main__":
    user_interface()
```

The script will:
- Load and preprocess data from the Excel file.
- Prompt for lag order selection (manual or via BIC/AIC).
- Ask for prior type and hyperparameter configuration.
- Handle exogenous variable forecasting (e.g., using last values, manual input, or ARIMA).
- Generate forecasts and display plots with historical data, median forecasts, and confidence intervals.

## File Structure
- `BVAR-X.py`: Main script containing all functions and the user interface.
- Data file (e.g., `Портфель.xlsx`): Excel file with financial data (not included; user must provide).

## Key Functions
- `minnesota_prior`: Implements the Litterman-Minnesota prior.
- `normal_flat_prior`: Implements a non-informative Normal-Flat prior.
- `normal_wishart_prior`: Implements the Normal-Wishart prior.
- `sims_zha_normal_flat_prior`: Implements the Sims-Zha Normal-Flat prior.
- `sims_zha_normal_wishart_prior`: Implements the Sims-Zha Normal-Wishart prior.
- `bvar_estimate`: Estimates the BVAR model with the specified prior and exogenous variables.
- `forecast_bvar`: Generates forecasts with confidence intervals.
- `plot_results`: Visualizes historical data, forecasts, and confidence intervals.
- `optimize_*_hyperparameters`: Optimizes hyperparameters for each prior type.
- `calculate_bic_aic`: Computes BIC or AIC for model selection.
- `preprocess_data`: Normalizes data (log-transformation and standardization).
- `inverse_transform_data`: Reverses normalization for interpretable results.
- `load_data_from_excel`: Loads and cleans data from an Excel file.
- `user_interface`: Provides an interactive console interface.

## Data Format
The Excel file should contain:
- **Columns**:
  - `Month`: Date in `DD.MM.YYYY` format.
  - `Uninvestedfunds`, `NettoFunds`, `Reinvestedfunds`, `Totalclients`: Endogenous variables.
  - `Investedfunds`, `Plannedrate`: Exogenous variables.
- **Rows**: Time series observations, with no missing values for the selected variables.

## Notes
- The script assumes the data is clean but handles minor formatting issues (e.g., spaces, non-breaking spaces).
- Exogenous variable forecasting requires sufficient data or user input for future values.
- The `statsmodels` library is optional and only required for ARIMA-based exogenous forecasting.
- Visualization uses `matplotlib` with a `ggplot` style for clear and aesthetic plots.
- The code includes error handling for numerical stability (e.g., ensuring positive semi-definite covariance matrices).

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for bug reports, feature requests, or suggestions.

## Contact
For questions or support, contact the repository maintainer or open an issue on GitHub.
