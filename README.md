# Stock Price Prediction Using LSTM Neural Networks

## Overview
This project focuses on predicting future stock prices using **Long Short-Term Memory (LSTM)** networks, a type of recurrent neural network (RNN) well-suited for time series forecasting. Historical stock price data is used to train the model, which then predicts the closing price of a stock for the next day.

## Objective
- **Preprocess historical stock price data** to ensure quality inputs for the model.
- **Build and train an LSTM model** to forecast future stock prices.
- **Evaluate the model's performance** and visualize predictions.

## Dataset
The dataset consists of historical stock price data downloaded using the **yfinance** library. The data includes:
- **Stock Symbol**: AAPL (Apple Inc.)
- **Time Period**: January 2015 to January 2023
- **Feature Used**: Closing price (Close)

The data is normalized and split into training and testing sets for model development.

## Methodology
### 1. Data Collection
- **Source**: Historical stock price data was downloaded using the **yfinance** library.
- **Symbol**: AAPL (Apple Inc.)
- **Period**: January 2015 to January 2023

### 2. Data Preprocessing
- **Normalization**: The closing price data was normalized to a range of **[0, 1]** using **MinMaxScaler** to improve model training efficiency.
- **Sequence Creation**: Sequences of 60 days were created to predict the next day's closing price.
- **Train-Test Split**: 80% of the data was used for training, and 20% was reserved for testing.

### 3. Model Architecture
- **Algorithm**: Long Short-Term Memory (LSTM)
- **Layers**:
  - **Input Layer**: Accepts sequences of 60 days.
  - **LSTM Layer**: Two stacked LSTM layers with 50 hidden units each.
  - **Fully Connected Layer**: Outputs the predicted stock price.
- **Activation Function**: ReLU (Rectified Linear Unit)
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam Optimizer

### 4. Training
- **Epochs**: 20
- **Batch Size**: 64
- **Cross-Validation**: Not explicitly used; instead, the model was evaluated on a separate test set.

### 5. Evaluation
- **Metrics**: Mean Squared Error (MSE) was used to evaluate the model's performance during training.
- **Visualization**: Actual vs. predicted stock prices were plotted to assess the model's accuracy.

## Results
The LSTM model achieved promising results in predicting stock prices:
- **Training Loss**: Reduced significantly over 20 epochs.
- **Prediction Accuracy**: The model closely followed the actual stock price trends, as observed in the visualization.

## Future Work
- **Further hyperparameter tuning** to improve model performance.
- **Incorporation of additional features** such as volume, moving averages, and technical indicators.
- **Testing with other deep learning architectures** such as GRU or Transformer-based models for better accuracy.

## How to Use
1. Ensure the dataset is available and formatted correctly.
2. Run the preprocessing script to prepare the data.
3. Train the LSTM model using the provided training script.
4. Evaluate the model and visualize predictions using the test dataset.

## Conclusion
This project demonstrates the use of **LSTM neural networks** for **time-series forecasting** in stock price prediction. With additional improvements and feature engineering, the model can be further optimized for financial market analysis.

