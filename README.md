# Comp-Sci--Assignment-Developing-a-Machine-Learning-App
The purpose of the GoldInsight Predictor is to analyze historical market data to predict the future closing price of Gold Futures. By analyzing past price trends and trading volume, the application aims to provide investors and analysts with a data-driven baseline for predicting short-term market movements, helping in the decision-making process.

Instructions for Running the App

1. Make sure you have Python (version 3.7 or higher) is installed on your machine.
2. Install the required dependencies by running the following command in your terminal:
3. pip install pandas numpy yfinance scikit-learn matplotlib seaborn
4. Save the source code provided in the assignment as a file named commodity_predictor.py.
5. Run the application using the command:
6. python commodity_predictor.py
7. The application will print performance metrics to the console and open a window displaying the prediction graph.


Discussion and Insights

Performance: The model achieved an R^2 score indicating how well the variance in the data is explained. (Insert your specific score here, usually high for time-series next-day prediction).
Limitations: One limitation observed was that the model didn’t have the ability to predict the record-breaking price surge in 2024. Because the Random Forest algorithm cannot extrapolate trends beyond the range of its training data, the predictions 'flatlined' near the historical maximums seen during training, while the actual price continued to rise.(as seen in Figure 1.0)
![alt text](<Untitled3 - Google Chrome 12_15_2025 9_00_48 PM.png>)



Improvements: Future iterations could include "Sentiment Analysis" from news headlines or include commodities often correlated with gold (like Silver or the US Dollar Index) as additional features to improve accuracy.
