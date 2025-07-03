import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import yfinance as yf
import pandas_ta as ta
import ssl
import urllib3
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.neural_network import MLPClassifier

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Create unverified SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# Custom MACD calculation function
def calculate_macd(prices, fast=12, slow=26, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line, macd - signal_line

# Custom RSI calculation function
# IF RSI > 70: overbought (possible price correction)
# IF RSI < 30: oversold (possible price increase)
def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100./(1.+rs)

    for i in range(period, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(period-1) + upval)/period
        down = (down*(period-1) + downval)/period
        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)
    return rsi

# Title
st.title('AutoML App (H2O Like Framework)')

# Sidebar navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Data Input', 'Model Training', 'Leaderboard', 'Model Details', 'Prediction', 'Trading Signals'])

# Session state for data and models
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'models' not in st.session_state:
    st.session_state['models'] = {}
if 'leaderboard' not in st.session_state:
    st.session_state['leaderboard'] = None
if 'best_model' not in st.session_state:
    st.session_state['best_model'] = None

# Data Input Page
if page == 'Data Input':
    st.header('Get Stock Data')
    
    data_source = st.radio(
        "Choose data source:",
        ('Download Historical Data', 'Upload CSV file')
    )
    
    if data_source == 'Download Historical Data':
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.text_input('Stock Symbol (e.g., AAPL)', 'AAPL')
        with col2:
            start_date = st.date_input('Start Date', pd.to_datetime('2020-01-01'))
        with col3:
            end_date = st.date_input('End Date', pd.to_datetime('2025-03-31'))
            
        if st.button('Download Data'):
            with st.spinner('Downloading and processing data...'):
                try:
                    # Download stock data with retry logic
                    max_retries = 5
                    retry_delay = 10  # seconds
                    
                    for attempt in range(max_retries):
                        try:
                            # Get the ticker object and verify it exists
                            ticker = yf.Ticker(symbol)
                            
                            # First verify the ticker exists and is valid
                            try:
                                info = ticker.info
                                if not info:
                                    st.error(f"Symbol {symbol} not found or may be delisted.")
                                    st.stop()
                                
                                # Display basic info
                                st.info(f"""
                                 Ticker: {symbol}
                                Current Price: ${info.get('currentPrice', 'N/A')}
                                Previous Close: ${info.get('previousClose', 'N/A')}
                                Day's Range: ${info.get('dayLow', 'N/A')} - ${info.get('dayHigh', 'N/A')}
                                Volume: {info.get('volume', 'N/A'):,}
                                Market Cap: ${info.get('marketCap', 'N/A'):,}
                                """)
                                
                            except Exception as e:
                                st.error(f"Error verifying symbol {symbol}: {str(e)}")
                                st.stop()
                            
                            # Get historical data with retry logic
                            max_retries = 5
                            retry_delay = 10  # seconds
                            
                            for attempt in range(max_retries):
                                try:
                                    stock_data = ticker.history(
                                        period="10y",  # Use 10 years of data for more samples
                                        interval="1d",
                                        auto_adjust=True
                                    )
                                    
                                    if stock_data is None or stock_data.empty:
                                        st.warning(f"Attempt {attempt + 1}: No historical data returned for {symbol}")
                                        if attempt < max_retries - 1:
                                            st.info(f"Retrying in {retry_delay} seconds...")
                                            time.sleep(retry_delay)
                                            continue
                                        else:
                                            st.error("Failed to download historical data. Please try again later.")
                                            st.stop()
                                    
                                    # If we got data, break the retry loop
                                    break
                                    
                                except Exception as e:
                                    if "Too Many Requests" in str(e):
                                        st.warning(f"Rate limit reached. Waiting {retry_delay} seconds before retry {attempt + 1}/{max_retries}...")
                                        time.sleep(retry_delay)
                                        continue
                                    else:
                                        st.error(f"Error downloading data: {str(e)}")
                                        st.stop()
                            
                            # Verify we have the required columns
                            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                            missing_columns = [col for col in required_columns if col not in stock_data.columns]
                            if missing_columns:
                                st.error(f"Missing required columns in downloaded data: {', '.join(missing_columns)}")
                                st.stop()
                            
                            # Calculate technical indicators
                            try:
                                # RSI
                                stock_data['RSI'] = calculate_rsi(stock_data['Close'].values)
                                
                                # MACD
                                macd, signal, hist = calculate_macd(stock_data['Close'])
                                stock_data['MACD'] = macd
                                stock_data['MACD_Signal'] = signal
                                stock_data['MACD_Hist'] = hist
                                
                                # Volatility
                                stock_data['Volatility'] = stock_data['Close'].pct_change().rolling(window=20).std()
                                
                                # Drop rows with NaN values
                                stock_data = stock_data.dropna()
                                
                                if stock_data.empty:
                                    st.error('No valid data after processing. Please try again.')
                                    st.stop()
                                
                                # Store data in session state
                                st.session_state['data'] = stock_data
                                
                                st.success(f' Downloaded {len(stock_data)} records of {symbol} stock data')
                                st.write('Data Preview:')
                                st.dataframe(stock_data.head())
                                
                                # Add download button
                                csv = stock_data.to_csv(index=True)
                                st.download_button(
                                    label="Download data as CSV",
                                    data=csv,
                                    file_name=f'{symbol}_stock_data.csv',
                                    mime='text/csv',
                                )
                                
                            except Exception as e:
                                st.error(f"Error processing the data: {str(e)}")
                                st.stop()
                                
                        except Exception as e:
                            if attempt == max_retries - 1:
                                st.error(f"Failed to download data after {max_retries} attempts.")
                                st.error(f"Error details: {str(e)}")
                                st.info("Troubleshooting tips:\n"
                                       "1. Verify the stock symbol is correct (e.g., MSFT for Microsoft)\n"
                                       "2. The stock might be temporarily unavailable\n"
                                       "3. Check your internet connection\n"
                                       "4. Try again in a few minutes as Yahoo Finance has rate limits")
                                st.stop()
                            st.warning(f"Attempt {attempt + 1} failed. Retrying in 5 seconds...")
                            time.sleep(5)  # Add delay between retries
                            continue
                    
                except Exception as e:
                    st.error(f'Error downloading data: {str(e)}')
                    st.info('Common issues:\n'
                           '1. Check if the stock symbol is correct\n'
                           '2. Try a different date range\n'
                           '3. Check your internet connection')
    
    else:
        st.header('Upload your CSV data')
        uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
        if uploaded_file:
            try:
                # Read CSV file with date parsing in a single step
                data = pd.read_csv(
                    uploaded_file,
                    parse_dates=['Date'],  # Explicitly parse the 'Date' column
                    index_col='Date'       # Set 'Date' as index
                )
                
                if data.empty:
                    st.error('File is empty or could not be read.')
                    st.stop()
                
                # Display file information
                st.write("File Information:")
                st.write(f"Number of columns: {len(data.columns)}")
                st.write(f"Columns: {', '.join(data.columns)}")
                st.write("First few rows:")
                st.dataframe(data.head())
                
                # Check if required columns exist
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_columns = [col for col in required_columns if col not in data.columns]
                if missing_columns:
                    st.error(f'Missing required columns: {", ".join(missing_columns)}')
                    st.info('Available columns:')
                    st.write(data.columns.tolist())
                    st.stop()
                
                # Remove existing signal column if it exists
                if 'Signal' in data.columns:
                    data = data.drop('Signal', axis=1)
                    st.info('Removed existing Signal column to generate new signals.')
                
                # Calculate technical indicators
                try:
                    # RSI
                    if 'RSI' not in data.columns:
                        data['RSI'] = calculate_rsi(data['Close'].values)
                    
                    # MACD
                    if 'MACD' not in data.columns or 'MACD_Signal' not in data.columns:
                        macd, signal, hist = calculate_macd(data['Close'])
                        data['MACD'] = macd
                        data['MACD_Signal'] = signal
                        data['MACD_Hist'] = hist
                    
                    # Volatility
                    if 'Volatility' not in data.columns:
                        data['Volatility'] = data['Close'].pct_change().rolling(window=20).std()
                    
                    # Create trading signals
                    data['Signal'] = 0  # Default to 0 (Hold)
                    
                    # Level 1: Match observed patterns
                    data.loc[(data['RSI'] < 35) & (data['MACD'] < data['MACD_Signal']), 'Signal'] = 1  # Buy
                    data.loc[(data['RSI'] > 65) & (data['MACD'] > data['MACD_Signal']), 'Signal'] = -1  # Sell
                    
                    # Display initial signal distribution
                    signal_counts = data['Signal'].value_counts()
                    st.subheader("Initial Signal Distribution")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Buy Signals", signal_counts.get(1, 0))
                    with col2:
                        st.metric("Hold Signals", signal_counts.get(0, 0))
                    with col3:
                        st.metric("Sell Signals", signal_counts.get(-1, 0))
                    
                    # Store data in session state
                    st.session_state['data'] = data
                    
                    st.success(f' Successfully processed {len(data)} records')
                    
                except Exception as e:
                    st.error(f'Error processing the data: {str(e)}')
                    st.stop()
                    
            except Exception as e:
                st.error(f'Error reading CSV file: {str(e)}')
                st.info('Please check that your file:')
                st.info('1. Is a valid CSV file')
                st.info('2. Has proper column headers')
                st.info('3. Contains the required columns: Open, High, Low, Close, Volume')
                st.stop()

# Model Training Page
elif page == 'Model Training':
    st.header(' Automated Model Training')
    
    if st.session_state['data'] is None:
        st.warning(' Please input data first in the Data Input page.')
    else:
        # Model selection dropdown
        model_options = {
            'RandomForest': 'Random Forest',
            'XGBoost': 'XGBoost',
            'LogisticRegression': 'Logistic Regression',
            'GradientBoosting': 'Gradient Boosting',
            'AdaBoost': 'AdaBoost',
            'NeuralNet': 'Neural Network (MLP)'
        }
        selected_models = st.multiselect(
            'Select models to train:',
            options=list(model_options.keys()),
            default=['RandomForest', 'XGBoost', 'LogisticRegression', 'GradientBoosting', 'AdaBoost', 'NeuralNet'],
            format_func=lambda x: model_options[x]
        )
        
        if st.button(' Start Training', type='primary'):
            with st.spinner('Preparing data...'):
                data = st.session_state['data']
                
                # Prepare features and target
                X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Volatility', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist']]
                y = data['Signal']
                y_mapped = y.map({-1: 0, 0: 1, 1: 2})
                
                # Verify we have both classes
                if len(y.unique()) < 2:
                    st.error(" Not enough signal classes for training. Please ensure you have both Buy and Hold/Sell signals.")
                    st.stop()
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Define models
            models = {}
            if 'RandomForest' in selected_models:
                models['RandomForest'] = RandomForestClassifier()
            if 'XGBoost' in selected_models:
                models['XGBoost'] = XGBClassifier()
            if 'LogisticRegression' in selected_models:
                models['LogisticRegression'] = LogisticRegression()
            if 'GradientBoosting' in selected_models:
                models['GradientBoosting'] = GradientBoostingClassifier()
            if 'AdaBoost' in selected_models:
                models['AdaBoost'] = AdaBoostClassifier()
            if 'NeuralNet' in selected_models:
                models['NeuralNet'] = MLPClassifier(max_iter=500, random_state=42)
            
            leaderboard = []
            total_models = len(models)
            
            for i, (name, model) in enumerate(models.items(), 1):
                try:
                    # Update progress
                    progress = i / total_models
                    progress_bar.progress(progress)
                    status_text.text(f'Training {model_options[name]}... ({i}/{total_models})')
                    
                    # Train model
                    if name == 'XGBoost':
                        model.fit(X, y_mapped)
                        y_pred = pd.Series(model.predict(X)).map({0: -1, 1: 0, 2: 1})
                    else:
                        model.fit(X, y)
                        y_pred = model.predict(X)
                    acc = accuracy_score(y, y_pred)
                    leaderboard.append({'Model': model_options[name], 'Accuracy': acc})
                    st.session_state['models'][name] = model
                    
                    # Show model completion
                    st.success(f'{model_options[name]} trained successfully! Accuracy: {acc:.2%}')
                    
                except Exception as e:
                    st.warning(f"Error training {model_options[name]}: {str(e)}")
                    continue

            if not leaderboard:
                st.error("No models were successfully trained. Please check your data.")
                st.stop()
            
            # Final progress update
            progress_bar.progress(1.0)
            status_text.text('✨ Training completed!')
            
            # Create a beautiful leaderboard display
            st.balloons()  # Celebration animation
            
            st.markdown("""
            <style>
            .leaderboard {
                background-color: #f0f2f6;
                padding: 20px;
                border-radius: 10px;
                margin-top: 20px;
            }
            .model-card {
                background-color: white;
                color: black;
                padding: 15px;
                border-radius: 5px;
                margin: 10px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .model-card h3 {
                color: black !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="leaderboard">', unsafe_allow_html=True)
            st.subheader('Model Leaderboard')
            
            leaderboard_df = pd.DataFrame(leaderboard).sort_values('Accuracy', ascending=False)
            st.session_state['leaderboard'] = leaderboard_df
            st.session_state['best_model'] = models[list(models.keys())[leaderboard_df.index[0]]]
            
            # Display each model's performance
            for _, row in leaderboard_df.iterrows():
                accuracy = row['Accuracy']
                model_name = row['Model']
                
                # Create a gradient color based on accuracy
                color = f"hsl({120 * accuracy}, 70%, 50%)"
                
                st.markdown(f"""
                <div class="model-card">
                    <h3>{model_name}</h3>
                    <div style="background-color: {color}; height: 20px; width: {accuracy*100}%; border-radius: 10px;"></div>
                    <p>Accuracy: {accuracy:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show best model
            best_model_name = leaderboard_df.iloc[0]['Model']
            best_accuracy = leaderboard_df.iloc[0]['Accuracy']
            
            st.success(f"""
             Best Model: **{best_model_name}**  
             Accuracy: **{best_accuracy:.2%}**
            """)
            
            # Add feature importance visualization for tree-based models
            if hasattr(st.session_state['best_model'], 'feature_importances_'):
                st.subheader('Feature Importance')
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': st.session_state['best_model'].feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = go.Figure(go.Bar(
                    x=feature_importance['Importance'],
                    y=feature_importance['Feature'],
                    orientation='h'
                ))
                
                fig.update_layout(
                    title='Feature Importance',
                    height=400,
                    showlegend=False,
                    xaxis_title="Importance",
                    yaxis_title="Feature"
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Leaderboard Page
elif page == 'Leaderboard':
    st.header('Model Leaderboard')
    if st.session_state['leaderboard'] is None:
        st.info('No models trained yet.')
    else:
        st.dataframe(st.session_state['leaderboard'])
        st.bar_chart(st.session_state['leaderboard'].set_index('Model')['Accuracy'])



# Model Details Page
elif page == 'Model Details':
    st.header('Model Details')
    if not st.session_state['models']:
        st.info('No models available.')
    else:
        selected_model = st.selectbox('Select Model', list(st.session_state['models'].keys()))
        model = st.session_state['models'][selected_model]
        
        if hasattr(model, 'feature_importances_'):
            st.subheader('Feature Importance')
            feature_importance = pd.DataFrame({
                'Feature': ['Open', 'High', 'Low', 'Close', 'Volume', 'Volatility', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist'],
                'Importance': model.feature_importances_
            })
            feature_importance = feature_importance.sort_values('Importance', ascending=False)
            st.bar_chart(feature_importance.set_index('Feature')['Importance'])




# Prediction Page
elif page == 'Prediction':
    st.header('Make Predictions')
    if st.session_state['best_model'] is None:
        st.info('No trained model available.')
    else:
        st.write('Using best performing model:', st.session_state['leaderboard'].iloc[0]['Model'])
        
        # Show signal generation rules
        st.subheader('Signal Generation Rules:')
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Buy Signal (1) when:**
            - RSI < 35 AND
            - MACD < MACD Signal
            
            Example Buy values:
            - RSI: 32
            - MACD: -1.5
            - MACD Signal: -1.2
            """)
        
        with col2:
            st.markdown("""
            **Sell Signal (-1) when:**
            - RSI > 65 AND
            - MACD > MACD Signal
            
            Example Sell values:
            - RSI: 68
            - MACD: 1.5
            - MACD Signal: 1.2
            """)
        
        # Allow user to input values for prediction
        st.subheader('Enter values for prediction:')
        col1, col2 = st.columns(2)
        
        with col1:
            open_price = st.number_input('Open Price', value=100.0)
            high_price = st.number_input('High Price', value=102.0)
            low_price = st.number_input('Low Price', value=98.0)
            close_price = st.number_input('Close Price', value=101.0)
            volume = st.number_input('Volume', value=1000000)
            
        with col2:
            volatility = st.number_input('Volatility', value=0.02, format="%.4f")
            rsi = st.number_input('RSI (30-70)', value=50.0, min_value=0.0, max_value=100.0)
            macd = st.number_input('MACD', value=0.0, format="%.4f")
            macd_signal = st.number_input('MACD Signal', value=0.0, format="%.4f")
            macd_hist = st.number_input('MACD Histogram', value=macd - macd_signal, format="%.4f", disabled=True)
            
        if st.button('Predict'):
            # Create input array for prediction
            X_pred = np.array([[open_price, high_price, low_price, close_price, volume, 
                              volatility, rsi, macd, macd_signal, macd_hist]])
            
            # Make prediction
            prediction = st.session_state['best_model'].predict(X_pred)[0]
            
            # Display result with explanation
            st.subheader('Prediction Result:')
            if prediction == 1:
                st.success('Prediction: BUY')
                st.markdown("""
                **Why Buy?**
                - RSI is in oversold territory (< 35)
                - MACD is below MACD Signal, indicating downward momentum may be ending
                """)
            elif prediction == -1:
                st.info('Prediction: SELL')
                st.markdown("""
                **Why Sell?**
                - RSI is in overbought territory (> 65)
                - MACD is above MACD Signal, indicating upward momentum may be ending
                """)
            else:
                st.info('Prediction: HOLD')
                st.markdown("""
                **Why Hold?**
                - RSI is in neutral territory (35-65)
                - Or MACD/Signal relationship doesn't confirm the trade
                """)
            
            # Show current values vs thresholds
            st.subheader('Current Values vs Thresholds:')
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RSI", f"{rsi:.2f}", 
                         f"{'Below' if rsi < 35 else 'Above' if rsi > 65 else 'Within'} threshold")
            with col2:
                st.metric("MACD vs Signal", f"{macd - macd_signal:.4f}", 
                         f"{'Below' if macd < macd_signal else 'Above'} signal line")




# Trading Signals Page
elif page == 'Trading Signals':
    st.header('Trading Signals Analysis')
    
    if st.session_state['data'] is None:
        st.warning('Please input data first in the Data Input page.')
    else:
        data = st.session_state['data'].copy()
        
        # Create trading signals
        data['Signal'] = 0  # Default to 0 (Hold)
        
        # Level 1: Match observed patterns
        data.loc[(data['RSI'] < 35) & (data['MACD'] < data['MACD_Signal']), 'Signal'] = 1  # Buy
        data.loc[(data['RSI'] > 65) & (data['MACD'] > data['MACD_Signal']), 'Signal'] = -1  # Sell
        
        # Display initial signal distribution
        signal_counts = data['Signal'].value_counts()
        st.subheader("Initial Signal Distribution")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Buy Signals", signal_counts.get(1, 0))
        with col2:
            st.metric("Hold Signals", signal_counts.get(0, 0))
        with col3:
            st.metric("Sell Signals", signal_counts.get(-1, 0))
        
        # Level 2: More lenient thresholds if needed
        min_signals = 5
        if signal_counts.get(1, 0) < min_signals or signal_counts.get(-1, 0) < min_signals:
            st.warning(" Not enough signals. Using more lenient thresholds...")
            
            # Reset signals
            data['Signal'] = 0
            
            # More lenient thresholds
            data.loc[(data['RSI'] < 40) & (data['MACD'] < data['MACD_Signal']), 'Signal'] = 1  # Buy
            data.loc[(data['RSI'] > 60) & (data['MACD'] > data['MACD_Signal']), 'Signal'] = -1  # Sell
            
            # Display adjusted signal distribution
            signal_counts = data['Signal'].value_counts()
            st.subheader("Adjusted Signal Distribution")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Buy Signals", signal_counts.get(1, 0))
            with col2:
                st.metric("Hold Signals", signal_counts.get(0, 0))
            with col3:
                st.metric("Sell Signals", signal_counts.get(-1, 0))
            
            # Level 3: Most lenient thresholds if still needed
            if signal_counts.get(1, 0) < min_signals or signal_counts.get(-1, 0) < min_signals:
                st.warning(" Still not enough signals. Using most lenient thresholds...")
                
                # Reset signals
                data['Signal'] = 0
                
                # Most lenient thresholds
                data.loc[(data['RSI'] < 45) & (data['MACD'] < data['MACD_Signal']), 'Signal'] = 1  # Buy
                data.loc[(data['RSI'] > 55) & (data['MACD'] > data['MACD_Signal']), 'Signal'] = -1  # Sell
                
                # Display final signal distribution
                signal_counts = data['Signal'].value_counts()
                st.subheader("Final Signal Distribution")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Buy Signals", signal_counts.get(1, 0))
                with col2:
                    st.metric("Hold Signals", signal_counts.get(0, 0))
                with col3:
                    st.metric("Sell Signals", signal_counts.get(-1, 0))
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Price Chart", "Technical Indicators", "Signal Distribution", "Signals Table"])
        
        with tab1:
            st.subheader("Stock Price with Trading Signals")
            
            # Create price chart with signals
            fig_price = go.Figure()
            
            # Add price line
            fig_price.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    name="Price",
                    line=dict(color='blue', width=2)
                )
            )
            
            # Add Buy signals
            buy_signals = data[data['Signal'] == 1]
            if not buy_signals.empty:
                fig_price.add_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['Close'],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color='green',
                            line=dict(width=2, color='black')
                        ),
                        name='Buy Signal'
                    )
                )
            
            # Add Sell signals
            sell_signals = data[data['Signal'] == -1]
            if not sell_signals.empty:
                fig_price.add_trace(
                    go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals['Close'],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color='red',
                            line=dict(width=2, color='black')
                        ),
                        name='Sell Signal'
                    )
                )
            
            # Update layout
            fig_price.update_layout(
                title="Stock Price with Trading Signals",
                height=500,
                showlegend=True,
                xaxis_title="Date",
                yaxis_title="Price ($)"
            )
            
            st.plotly_chart(fig_price, use_container_width=True)
        
        with tab2:
            st.subheader("Technical Indicators")
            
            # Create figure for technical indicators
            fig_tech = go.Figure()
            
            # Add RSI
            fig_tech.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    name="RSI",
                    line=dict(color='purple', width=1)
                )
            )
            
            # Add MACD
            fig_tech.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD'],
                    name="MACD",
                    line=dict(color='blue', width=1)
                )
            )
            
            # Add MACD Signal
            fig_tech.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD_Signal'],
                    name="MACD Signal",
                    line=dict(color='red', width=1)
                )
            )
            
            # Update layout
            fig_tech.update_layout(
                title="Technical Indicators",
                height=500,
                showlegend=True,
                xaxis_title="Date",
                yaxis_title="Value"
            )
            
            st.plotly_chart(fig_tech, use_container_width=True)


            
        with tab3:
            st.subheader("Signal Distribution")
            
            # Create pie chart
            fig_pie = go.Figure()
            fig_pie.add_trace(
                go.Pie(
                    labels=['Buy', 'Sell', 'Hold'],
                    values=[signal_counts.get(1, 0), signal_counts.get(-1, 0), signal_counts.get(0, 0)],
                    textinfo='label+percent',
                    marker=dict(colors=['green', 'red', 'gray'])
                )
            )
            
            # Update layout
            fig_pie.update_layout(
                title="Distribution of Trading Signals",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)





        with tab4:
            st.subheader("Trading Signals Table")
            
            # Create signals table
            signals_df = data[data['Signal'] != 0].copy()
            if not signals_df.empty:
                # Map signal values to text
                signals_df['Signal_Type'] = signals_df['Signal'].map({1: 'Buy', -1: 'Sell'})
                
                # Keep the index (which is the date) and sort by it
                signals_df = signals_df.sort_index(ascending=False)
                
                # Format the DataFrame for display
                display_df = signals_df[['Signal_Type', 'Close', 'RSI', 'MACD', 'MACD_Signal']].copy()
                display_df['Close'] = display_df['Close'].round(2)
                display_df['RSI'] = display_df['RSI'].round(2)
                display_df['MACD'] = display_df['MACD'].round(4)
                display_df['MACD_Signal'] = display_df['MACD_Signal'].round(4)
                
                # Display the table
                st.dataframe(
                    display_df,
                    column_config={
                        "Signal_Type": st.column_config.TextColumn("Signal Type"),
                        "Close": st.column_config.NumberColumn("Price", format="$%.2f"),
                        "RSI": st.column_config.NumberColumn("RSI", format="%.2f"),
                        "MACD": st.column_config.NumberColumn("MACD", format="%.4f"),
                        "MACD_Signal": st.column_config.NumberColumn("MACD Signal", format="%.4f")
                    },
                    use_container_width=True
                )
                
                # Add download button for signals
                csv = signals_df.to_csv()
                st.download_button(
                    label="Download Signals as CSV",
                    data=csv,
                    file_name='trading_signals.csv',
                    mime='text/csv',
                )
            else:
                st.info("No trading signals found in the data.") 