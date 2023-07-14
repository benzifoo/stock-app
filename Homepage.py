pip install yfinance

import streamlit as st

st.set_page_config(page_title="Homepage")

st.title("Main Page")
st.sidebar.success("Select a page above.")

from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import plotly as px

from PIL import Image

from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric


def add_logo(logo_path, width, height):
    logo = Image.open("Logo.png")
    modified_logo = logo.resize((width, height))
    return modified_logo

my_logo = add_logo(logo_path="Logo.png", width=300, height=300)
st.sidebar.image(my_logo)


START = "2012-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction Application 📈")

image = Image.open('Image.png')
st.image(image)

st.write('This application enable stock price prediction. Any stocks as you like')
st.markdown("""Predictive model based on algorithm **[Prophet](https://facebook.github.io/prophet/)**.""")


stocks = ("AAPL", "GOOG", "MSFT", "GME", "7113.KL", "GC=F")
if 'selected_stock' not in st.session_state:
    st.session_state['selected_stock'] = 'AAPL'

if 'n_years' not in st.session_state:
    st.session_state['n_years'] = 1

st.session_state['selected_stock'] = st.sidebar.selectbox("Select dataset for prediction", stocks, index=stocks.index(st.session_state['selected_stock']))
st.session_state['n_years'] = st.sidebar.slider("Years of prediction:", 1 , 4, value=st.session_state['n_years'])
st.session_state['period'] = st.session_state['n_years'] * 365

selected_stock = st.session_state['selected_stock']
period = st.session_state['period']
n_years = st.session_state['n_years']

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data



data_load_state = st.sidebar.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")        

st.subheader('1. Data Exploratory')
st.markdown('It consists of tail of raw data and descriptive statistics')

st.subheader('Raw Data')
st.table(data.tail())

st.subheader('Descriptive Statistics')
st.markdown('It contains count, average, standard deviation, minimum, maximum and interquatile range')
st.table(data.describe())




st.subheader('Line Chart - 10 Years Overview')
st.markdown('Adjust slicer to have better view')

# def plot_raw_data():
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x = data['Date'], y = data['Open'], name = 'Open',
#     line = dict(color='royalblue')))
#     fig.add_trace(go.Scatter(x = data['Date'], y = data['Close'], name = 'Close',
#     line = dict(color='firebrick')))
#     fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible = True,
#     height = 600, yaxis_title='USD per Share')
#     st.plotly_chart(fig, use_container_width=True)

# plot_raw_data()

def plot_candle_data():
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name = 'Market Movement'))
    fig.update_layout(
    title='Share Price Movement',
    yaxis_title='USD per Share',
    yaxis=dict(tickprefix='$'),
    showlegend = False,
    height = 700)
    st.plotly_chart(fig, use_container_width=True)

plot_candle_data()


#Returns
data_returns = data[['Date', 'Adj Close']]
data_returns.rename(columns={'Adj Close' : 'price_t'}, inplace = True)
data_returns['returns'] = (data_returns['price_t'] / data_returns['price_t'].shift(1)) - 1

def plot_returns_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = data_returns['Date'], y = data_returns['returns'], hovertemplate='Returns: %{y:.2%}<extra></extra>'))
    fig.layout.update(title_text="Stock Returns", xaxis_rangeslider_visible = True,
    height = 600, yaxis_title='Returns Percentage %')
    st.plotly_chart(fig, use_container_width=True)

plot_returns_data()


st.subheader('EDA')

data.reset_index(inplace=True)
data.set_index('Date', inplace=True)


filtered_data = data.loc[START:TODAY]
def plot_ma_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Close'], name='Close', line=dict(color='black'), hovertemplate='Close: $%{y:.2f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Close'].rolling(window=50).mean(), name='50-day MA', line=dict(color='red'), hovertemplate='50-day MA: $%{y:.2f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Close'].rolling(window=200).mean(), name='200-day MA', line=dict(color='blue'), hovertemplate='200-day MA: $%{y:.2f}<extra></extra>'))
    fig.update_layout(title='Moving Averages', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True)

plot_ma_data()


def plot_volatility_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'].pct_change().rolling(window=21).std(), name='Volatility', hovertemplate='Date: %{x}<br>Volatility: %{y:.2%}<extra></extra>'))
    fig.update_layout(title='Volatility - Rolling standard deviation of the stock daily percentage change',
                  xaxis_title='Date',
                  yaxis_title='Volatility',
                  xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

plot_volatility_data()


import calendar

def plot_yearlyavg_data():
    # Get the unique years in the data
    years = data.index.year.unique()

    # Create a selectbox to filter by year
    selected_year = st.selectbox('Select Year', years)

    # Filter the DataFrame for the selected year
    filtered_data = data[data.index.year == selected_year]

    # Calculate monthly statistics: min, max, and average closing price for each month
    monthly_stats = filtered_data.groupby(filtered_data.index.month)['Close'].agg(['min', 'max', 'mean'])

    fig = go.Figure()

    # Iterate over each month
    for month in range(1, 13):
        # Check if the month exists in the monthly_stats DataFrame
        if month in monthly_stats.index:
            # Filter data for the specific month
            monthly_data = monthly_stats.loc[month]

            # Create a box plot for the month
            fig.add_trace(go.Box(
                y=monthly_data,
                name=calendar.month_name[month],
                boxpoints='all',
                jitter=0.5,
                whiskerwidth=0.2,
                marker=dict(size=2),
                line=dict(width=1),
                hovertemplate='Value: $%{y:.2f}<extra></extra>'
            ))

    fig.update_layout(
        title=f'Seasonal Analysis - Monthly Closing Price Statistics ({selected_year})',
        xaxis_title='Month',
        yaxis_title='Price'
    )

    # Render the figure using Streamlit
    st.plotly_chart(fig, use_container_width=True)

plot_yearlyavg_data()


# ##########Default has no value
# def plot_yearlyavg_data():
#     # Get the unique years in the data
#     years = data.index.year.unique()

#     # Create a multiselect to filter by year
#     selected_years = st.multiselect('Select Years', years)

#     # Filter the DataFrame for the selected years
#     filtered_data = data[data.index.year.isin(selected_years)]

#     # Calculate monthly statistics: min, max, and average closing price for each month
#     monthly_stats = filtered_data.groupby(filtered_data.index.month)['Close'].agg(['min', 'max', 'mean'])

#     fig = go.Figure()

#     # Iterate over each month
#     for month in range(1, 13):
#         # Check if the month exists in the monthly_stats DataFrame
#         if month in monthly_stats.index:
#             # Filter data for the specific month
#             monthly_data = monthly_stats.loc[month]

#             # Create a box plot for the month
#             fig.add_trace(go.Box(
#                 y=monthly_data,
#                 name=calendar.month_name[month],
#                 boxpoints='all',
#                 jitter=0.5,
#                 whiskerwidth=0.2,
#                 marker=dict(size=2),
#                 line=dict(width=1),
#                 hovertemplate='Value: $%{y:.2f}<extra></extra>'
#             ))

#     fig.update_layout(
#         title='Seasonal Analysis - Monthly Closing Price Statistics',
#         xaxis_title='Month',
#         yaxis_title='Price'
#     )

#     # Render the figure using Streamlit
#     st.plotly_chart(fig, use_container_width=True)

# plot_yearlyavg_data()


###########Default select all year
def plot_yearlyavg_data():
    # Get the unique years in the data
    years = data.index.year.unique()

    # Create a multiselect to filter by year, with all years selected by default
    selected_years = st.multiselect('Select Years', options=list(years), default=list(years))

    # Filter the DataFrame for the selected years
    filtered_data = data[data.index.year.isin(selected_years)]

    # Calculate monthly statistics: min, max, and average closing price for each month
    monthly_stats = filtered_data.groupby(filtered_data.index.month)['Close'].agg(['min', 'max', 'mean'])

    fig = go.Figure()

    # Iterate over each month
    for month in range(1, 13):
        # Check if the month exists in the monthly_stats DataFrame
        if month in monthly_stats.index:
            # Filter data for the specific month
            monthly_data = monthly_stats.loc[month]

            # Create a box plot for the month
            fig.add_trace(go.Box(
                y=monthly_data,
                name=calendar.month_name[month],
                boxpoints='all',
                jitter=0.5,
                whiskerwidth=0.2,
                marker=dict(size=2),
                line=dict(width=1),
                hovertemplate='Value: $%{y:.2f}<extra></extra>'
            ))

    fig.update_layout(
        title='Seasonal Analysis - Monthly Closing Price Statistics',
        xaxis_title='Month',
        yaxis_title='Price'
    )

    # Render the figure using Streamlit
    st.plotly_chart(fig, use_container_width=True)

plot_yearlyavg_data()



data['Return'] = data['Close'].pct_change()

# def plot_distribution_data():
#     fig = go.Figure()
#     fig.add_trace(
#         go.Histogram(
#             x=data['Return'],
#             marker=dict(
#                 color='#3498db',  # Use a modern blue color
#                 line=dict(
#                     color='black',
#                     width=1
#                 )
#             ),
#             hoverinfo='x+y',
#             hovertemplate="Return: %{x:.2%} <br>Count: %{y}",
#             name='Return Distribution'
#         )
#     )
#     fig.update_layout(
#         title='Price Change Distribution',
#         xaxis_title='Return',
#         yaxis_title='Frequency',
#         bargap=0.1,
#         hovermode='x unified',
#         plot_bgcolor='white',  # Set the plot background color to white
#         paper_bgcolor='white',  # Set the paper background color to white
#         font=dict(color='black')  # Set the font color to black
#     )
#     st.plotly_chart(fig, use_container_width=True)

# plot_distribution_data()


def plot_pairplot_data():
    # Define subplot structure
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=('Close vs Volume', 'Close vs Return', 'Volume vs Return'),
                        vertical_spacing=0.12, 
                        horizontal_spacing=0.12)

    # Close vs Volume
    fig.add_trace(
        go.Scattergl(x=data['Close'], y=data['Volume'], mode='markers', 
                    marker=dict(color='rgb(0, 0, 255)', 
                                line=dict(width=1, color='DarkSlateGrey')),
                    hovertemplate='Close: $%{x}<br>Volume: %{y}'),
        row=1, col=1
    )

    # Close vs Return
    fig.add_trace(
        go.Scattergl(x=data['Close'], y=data['Return'], mode='markers', 
                    marker=dict(color='rgb(127, 127, 127)', 
                                line=dict(width=1)),
                    hovertemplate='Close: $%{x}<br>Return: %{y:.2%}'),
        row=1, col=2
    )

    # Volume vs Return
    fig.add_trace(
        go.Scattergl(x=data['Volume'], y=data['Return'], mode='markers', 
                    marker=dict(color='rgb(255, 180, 0)', 
                                line=dict(width=1)),
                    hovertemplate='Volume: %{x}<br>Return: %{y:.2%}'),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(height=800, width=800, title_text="Pairplot")
    st.plotly_chart(fig, use_container_width=True)

plot_pairplot_data()





def plot_pairplot_data():
    # Define subplot structure
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=('<b>Price vs Volume</b>', '<b>Price vs Return</b>', '<b>Volume vs Return</b>'),
                        vertical_spacing=0.12, 
                        horizontal_spacing=0.12)

    # Price vs Volume
    fig.add_trace(
        go.Scattergl(x=data['Close'], y=data['Volume'], mode='markers', 
                    marker=dict(color='rgb(0, 0, 255)', 
                                line=dict(width=1, color='DarkSlateGrey')),
                    hovertemplate='Price: $%{x}<br>Volume: %{y}'),
        row=1, col=1
    )
    fig.update_xaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Volume', row=1, col=1)

    # Price vs Return
    fig.add_trace(
        go.Scattergl(x=data['Close'], y=data['Return'], mode='markers', 
                    marker=dict(color='rgb(127, 127, 127)', 
                                line=dict(width=1)),
                    hovertemplate='Price: $%{x}<br>Return: %{y:.2%}'),
        row=1, col=2
    )
    fig.update_xaxes(title_text='Price', row=1, col=2)
    fig.update_yaxes(title_text='Return', row=1, col=2)

    # Volume vs Return
    fig.add_trace(
        go.Scattergl(x=data['Volume'], y=data['Return'], mode='markers', 
                    marker=dict(color='rgb(255, 180, 0)', 
                                line=dict(width=1)),
                    hovertemplate='Volume: %{x}<br>Return: %{y:.2%}'),
        row=2, col=1
    )
    fig.update_xaxes(title_text='Volume', row=2, col=1)
    fig.update_yaxes(title_text='Return', row=2, col=1)

    # Update layout
    fig.update_layout(height=800, width=800, title_text="Pairplot")
    st.plotly_chart(fig, use_container_width=True)

plot_pairplot_data()
