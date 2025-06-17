# 📈 Stock Price Prediction with LSTM

A Jupyter notebook for time series forecasting using an LSTM neural network. It leverages historical Apple stock data from Yahoo Finance and builds a deep learning model to predict future prices.

---

## ⚙️ Requirements

Install required dependencies:

```bash
pip install yfinance matplotlib pandas numpy torch scikit-learn
```

---

## 📁 File

* `main.ipynb` – Jupyter notebook that:

  * Downloads historical stock data for a given ticker (default: `AAPL`)
  * Normalizes and sequences the data
  * Trains an LSTM model using PyTorch
  * Visualizes predicted vs actual stock prices

---

## 🚀 Usage

1. Launch the notebook:

   ```bash
   jupyter notebook main.ipynb
   ```

2. Run the cells step by step:

   * Download stock data
   * Preprocess and scale it
   * Train LSTM model
   * Visualize prediction results

---

## 🧠 Model

* Architecture: 2-layer LSTM with 32 hidden units
* Optimizer: Adam (`lr=0.01`)
* Loss: Mean Squared Error
* Sequence length: 30 days

---

## ✅ Example Output

![Sample Output Plot](example_plot.png)
*Prediction (green) vs actual (blue) prices on test data*

---

## 📊 Evaluation

The model reports RMSE (Root Mean Squared Error) for both training and test sets.