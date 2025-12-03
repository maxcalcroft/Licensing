import os
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import MetaTrader5 as mt5
from   datetime import datetime, timedelta
import streamlit_antd_components as sac
import smtplib

# Dict
deal_desc = {
   0: "Buy",
   1: "Sell",
   2: "Buy Limit",
   3: "Sell Limit",
   4: "Buy Stop",
   5: "Sell Stop"
}

#---------------------------------------------------------------#
# PAGE CONFIG                                                   #
#---------------------------------------------------------------#
st.set_page_config(
    page_title="Portfolio Analysis",
    page_icon=":material/finance_mode:",
    layout="wide",
    initial_sidebar_state="expanded",
)
#---------------------------------------------------------------#
# HEADER                                                        #
#---------------------------------------------------------------#
st.markdown(''' # **Portfolio Analysis**''')

#---------------------------------------------------------------#
# DIRECTORY AND PATH CHECKING                                   #
#---------------------------------------------------------------#
mt5_paths = {}

# Search in Program Files for terminal.exe files
if os.path.exists("C:/Program Files/"):
   program_files = os.listdir("C:/Program Files/")

   for folder in program_files:
      try:
         folder_path = f"C:/Program Files/{folder}"
         if not os.path.isdir(folder_path):
            continue
         folder_files = os.listdir(folder_path)
         if "terminal64.exe" in folder_files:
            mt5_path = f"{folder_path}/terminal64.exe"
            mt5_paths[folder] = mt5_path
      except (PermissionError, OSError):
         continue

#---------------------------------------------------------------#
# SIDE BAR                                                      #
#---------------------------------------------------------------#
with st.sidebar:

   items = []
   for folder in mt5_paths:
      items.append(sac.SegmentedItem(label=folder, icon='user'))

   # Add default item if no accounts found
   if not items:
      items.append(sac.SegmentedItem(label="No Account", icon='graph-up'))

   selected_account = sac.segmented(items=items, label="Accounts", align='center', use_container_width=True, direction='vertical', color="#2178cfff", return_index=True)
   
   sac.divider(label='Details', align='center', size='sm', color='#2178cfff')
   login_placeholder = st.empty()
   name_placeholder = st.empty()
   company_placeholder = st.empty()
   server_placeholder = st.empty()
   currency_placeholder = st.empty()
   leverage_placeholder = st.empty()

#---------------------------------------------------------------#
# Check Terminal Connections                                    #
#---------------------------------------------------------------#
@st.fragment(run_every=600)
def CheckConnections():
   for key in mt5_paths.keys():
      if mt5.initialize(mt5_paths[key]) and not mt5.terminal_info().connected:

         smtp_server = "smtp.gmail.com"
         smtp_port = 587
         sender_email = "maxcalcroft96@gmail.com"
         sender_password = "mxnqzvyilwkfarzi"
         recipient_email = "maxcalcroft96@gmail.com"
         account_name = key

         subject = f"MT5 Connection Lost - {account_name}"
         body = f"""MT5 Connection Alert

         Account: {account_name}
         Account Login: {mt5.account_info().login}
         Status: Disconnected
         Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
         Error: {mt5.last_error()}

         Please check your MT5 terminal and reconnect.
         """
        
         message = f"Subject: {subject}\n\n{body}"
         
         # Send email
         with smtplib.SMTP(smtp_server, smtp_port) as server:
               server.starttls()
               server.login(sender_email, sender_password)
               server.sendmail(sender_email, recipient_email, message)
      mt5.shutdown()
   return True
            
CheckConnections()
#---------------------------------------------------------------#
# ACCOUNT SELECTION AND MT5 LOGIN                               #
#---------------------------------------------------------------#
login_display = st.empty()

# Auto login to MT5 (selected account returns an index, path extracted via list casting)
try:
   if mt5.initialize((list(mt5_paths.values()))[selected_account]) and mt5.terminal_info().connected: 

      sac.alert(label="Connected to:", description=f"{mt5.account_info().company}  |  Account: {mt5.account_info().login}", radius='sm', variant='light', color='success', banner=[False, True], icon="bar-chart-fill", closable=False)

      account_info = mt5.account_info()
      login_placeholder.markdown(f"**Login:** {account_info.login}")
      name_placeholder.markdown(f"**Name:** {account_info.name}")
      company_placeholder.markdown(f"**Company:** {account_info.company}")
      server_placeholder.markdown(f"**Server:** {account_info.server}")
      currency_placeholder.markdown(f"**Currency:** {account_info.currency}")
      leverage_placeholder.markdown(f"**Leverage:** 1:{account_info.leverage}")

   else:
      login_display.error(f"MT5 initialization failed, error code: {mt5.last_error()}")

except Exception as e:
   login_display.error(f"Failed to connect to MetaTrader 5: {e}")

#---------------------------------------------------------------#
# CACHED DEALS DATA FUNCTION                                    #
#---------------------------------------------------------------#

def get_cached_deals_data(account_login):
   """
   Fetch and cache deals data from MT5 to avoid repeated API calls.
   Returns a pandas DataFrame with all deals data.
   Args:
       account_login: The account login number to use as cache key
   """
   from_date = datetime(2023, 1, 1)
   to_date = datetime.now() + timedelta(minutes=1400)
   
   deals = mt5.history_deals_get(from_date, to_date)
    
   if deals is not None and len(deals) > 0:
      # Convert to DataFrame and add time conversion
      df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
      df['time'] = pd.to_datetime(df['time'], unit='s')
      
      # Calculate initial balance for this account if not already cached
      if f'initial_balance_{account_login}' not in st.session_state:
          balance_deals = [deal for deal in deals if deal.type == mt5.DEAL_TYPE_BALANCE]
          st.session_state[f'initial_balance_{account_login}'] = sum(deal.profit for deal in balance_deals)
      
      # Set current account's initial balance
      st.session_state.initial_balance = st.session_state[f'initial_balance_{account_login}']
 
      df = df[(df['type'] != mt5.DEAL_TYPE_BALANCE) & (df['type'] != mt5.DEAL_TYPE_CREDIT) & (df['type'] != mt5.DEAL_TYPE_CHARGE)]

      return df
   else:
      # Return empty DataFrame with expected columns if no deals
      return pd.DataFrame(columns=['time', 'swap', 'profit', 'commission', 'magic', 'entry', 'type', 'comment', 'symbol', 'volume'])

get_cached_deals_data(mt5.account_info().login if mt5.account_info() else 0)

#---------------------------------------------------------------#
# TIME/PROFIT/BALANCE FRAGMENT UPDATER                          #
#---------------------------------------------------------------#
# Hide metrics toggle
if 'hide_metrics' not in st.session_state:
   st.session_state.hide_metrics = True

st.toggle("Hide Live Metrics ", on_change=lambda: setattr(st.session_state, "hide_metrics", not st.session_state.hide_metrics),
         help="Toggle to hide sensitive account metrics like balance and profit",
         value=True)
   
# Create placeholders for metrics OUTSIDE the fragment
time_col, PnL_col, Balance_col, WeeklyBalance_col, MonthlyBalance_col = st.columns(5)

with time_col:
   time_placeholder = st.empty()
with PnL_col:
   pnl_placeholder = st.empty()
with Balance_col:
   balance_placeholder = st.empty()
with WeeklyBalance_col:
   weekly_placeholder = st.empty()
with MonthlyBalance_col:
   monthly_placeholder = st.empty()


@st.fragment(run_every=1) 
def update_live_metrics():
   account_info = mt5.account_info()
   balance = account_info.balance if account_info else 0
   profit = account_info.profit if account_info else 0

   # Update ONLY the content, not the structure
   current_time = pd.to_datetime('now').strftime('%H:%M:%S')
   time_placeholder.metric(label="Time", value=current_time)
   
   pnl_placeholder.metric(label="Unrealized P&L", value=f"{profit:.2f}" if st.session_state.hide_metrics == False else " - - ",
              delta=f"{(profit / balance * 100) if balance != 0 else 0:.2f}%" if st.session_state.hide_metrics == False else None, delta_color="normal")
   
   # Safely get initial balance with fallback
   initial_balance = st.session_state.get('initial_balance', 0)
   delta_pct = ((balance - initial_balance) / initial_balance * 100) if initial_balance != 0 else 0

   balance_placeholder.metric(label="Account Balance", value=f"{balance:.2f}" if st.session_state.hide_metrics == False else " - - ",
              delta=f"{delta_pct:.2f}%" if st.session_state.hide_metrics == False else None, delta_color="normal")
   
   # Get start of current week (Monday) using cached deals
   start_of_week = datetime.now() - timedelta(days=datetime.now().weekday())
   start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)

   # Use cached deals for weekly profit calculation
   account_info = mt5.account_info()
   current_login = account_info.login if account_info else 0
   cached_deals = get_cached_deals_data(current_login)
   
   if not cached_deals.empty:
      week_deals_filtered = cached_deals[
         (cached_deals['time'] >= start_of_week)
      ]
      week_profit = week_deals_filtered['profit'].sum() if not week_deals_filtered.empty else 0
   else:
       week_profit = 0

   weekly_placeholder.metric(label="Weekly Profit", value=f"{week_profit:.2f}" if st.session_state.hide_metrics == False else " - - ",
             delta=f"{(week_profit / balance * 100) if balance != 0 else 0:.2f}%" if st.session_state.hide_metrics == False else None, delta_color="normal")
   
   # Get start of current month using cached deals
   start_of_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)

   # Use cached deals for monthly profit calculation
   if not cached_deals.empty:
       month_deals_filtered = cached_deals[
           
           (cached_deals['time'] >= start_of_month)
       ]
       month_profit = month_deals_filtered['profit'].sum() if not month_deals_filtered.empty else 0
   else:
       month_profit = 0

   monthly_placeholder.metric(label="Monthly Profit", value=f"{month_profit:.2f}" if st.session_state.hide_metrics == False else " - - ",
             delta=f"{(month_profit / balance * 100) if balance != 0 else 0:.2f}%" if st.session_state.hide_metrics == False else None, delta_color="normal")



update_live_metrics()

#---------------------------------------------------------------#
# LIVE METRICS TABS                                             #
#---------------------------------------------------------------#

tabs = ['Live Metrics', 'Performance', 'Correlation', 'Trade History']

tabs = st.tabs(tabs)

live_metrics = tabs[0]
performance_metrics = tabs[1]
heatmap = tabs[2]
history = tabs[3]

css = '''
<style>
   .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
   font-size:1.1rem;
   }
</style>
'''

st.markdown(css, unsafe_allow_html=True)
with live_metrics:
   # Create placeholders OUTSIDE the fragment to prevent recreation
   positions_header = st.empty()
   positions_container = st.empty()
   orders_header = st.empty()
   orders_container = st.empty()
   
   @st.fragment(run_every=1)  
   def update_positions():

      positions_header.write("Open Positions")
      
      # Clear the list each time to avoid accumulation
      positions_data = []
      positions = mt5.positions_total()
      
      if positions is not None and positions > 0:
         # Get all positions at once instead of iterating by index
         all_positions = mt5.positions_get()
         if all_positions is not None:
            for position in all_positions:  # Iterate directly over positions
                  positions_data.append({
                    "Symbol": position.symbol,
                    "Magic": position.magic,
                    "Open Time": pd.to_datetime(position.time, unit='s').strftime('%Y-%m-%d %H:%M:%S'),
                    "Type": deal_desc.get(position.type, "Unknown"),
                    "Volume": f"{position.volume:.2f}",
                    "Swap": f"{position.swap:.2f}",
                    "Profit": f"{position.profit:.2f}" if st.session_state.hide_metrics == False else " - - ",
                    "Comment": position.comment,
                  })
            
            # Display as a DataFrame using the placeholder
            if positions_data:  # Only create DataFrame if we have data
               df = pd.DataFrame(positions_data)
               positions_container.dataframe(df, width='stretch', hide_index=True)
   
         else:
            positions_container.error("Failed to retrieve positions")
      else:
         positions_container.info("No open positions")
      
      orders_header.write("Open Orders")
      orders_data = []
      orders = mt5.orders_total()
      
      if orders is not None and orders > 0:
         all_orders = mt5.orders_get()
         if all_orders is not None:
            for order in all_orders:
               # Get digits of symbol
               digits = mt5.symbol_info(order.symbol).digits

               orders_data.append({
                  "Symbol": order.symbol,
                  "Magic": order.magic,
                  "Order Time": pd.to_datetime(order.time_setup, unit='s').strftime('%Y-%m-%d %H:%M:%S'),
                  "Type": deal_desc.get(order.type, "Unknown"),
                  "Volume": f"{order.volume_current:.2f}",
                  "Price": f"{order.price_open:.{digits}f}",
                  "Comment": order.comment,
               })

            if orders_data:
               df_orders = pd.DataFrame(orders_data)
               orders_container.dataframe(df_orders, width='stretch', hide_index=True)
            else:
               orders_container.info("No order data available")
         else:
            orders_container.error("Failed to retrieve orders")
     
   update_positions()

   sac.divider(label='Portfolio Equity', icon=sac.BsIcon(name='graph-up', size=10), align='center', size='md', color='#f5f5dc')


   # Create placeholder for equity chart OUTSIDE fragment
   equity_chart_placeholder = st.empty()

   @st.fragment(run_every=60)
   def graph_update():
      # Use cached deals data with account-specific cache
      account_info = mt5.account_info()
      current_login = account_info.login if account_info else 0
      cached_deals = get_cached_deals_data(current_login)
  
      if not cached_deals.empty:
          
         # Safely get initial balance with fallback
         initial_balance = getattr(st.session_state, 'initial_balance', 0)
         if initial_balance == 0:
             account_info = mt5.account_info()
             initial_balance = account_info.balance if account_info else 0
                     
         # Create cumulative profit
         cached_deals['total_profit'] = (
             initial_balance + np.cumsum(
                 cached_deals['profit'].round(2) +
                 cached_deals['swap'].round(2) +
                 cached_deals['commission'].round(2)
             )
         )
         
         # Create the chart
         fig = px.line(cached_deals, x='time', y='total_profit')
                      
         fig.update_layout(xaxis_title='Time', 
                           yaxis_title='Total Profit',
                           template='ggplot2')
      else:
         # Create empty chart if no filtered data
         fig = px.line(title='Portfolio Equity')
         fig.update_layout(xaxis_title='Time', 
                           yaxis_title='Total Profit',
                           template='ggplot2')

      # Update ONLY the chart content using placeholder
      equity_chart_placeholder.plotly_chart(fig, config={'responsive': True})
      fig.update_layout(height=800)
   
   graph_update()

#---------------------------------------------------------------#
# Performance Metrics                                           #
#---------------------------------------------------------------#

with performance_metrics:

   #---------------------------------------------------------------#
   # Get History Deal Data from Cache

   account_info = mt5.account_info()
   current_login = account_info.login if account_info else 0
   cached_deals = get_cached_deals_data(current_login)

   #---------------------------------------------------------------#
   # Filter data by magic and plot profit by strategy
   strat_df = pd.DataFrame()
   if cached_deals.empty:
      st.error("No deals data available")
      
   else:
      df_charts_filtered = cached_deals.copy()
      
      # Keep only the specified columns
      df_charts_filtered = df_charts_filtered[['symbol', 'time', 'swap', 'entry', 'type', 'profit', 'commission', 'comment', 'magic']].copy()
      
      # Group by magic number and calculate cumulative profit for each strategy
      strategy_data = []
      
      for magic in df_charts_filtered['magic'].unique():
         magic_data = df_charts_filtered[df_charts_filtered['magic'] == magic].copy()
         magic_data = magic_data.sort_values('time')  # Ensure chronological order
         
         # Calculate cumulative profit for this magic number
         magic_data['cumulative_profit'] = (
           magic_data['profit'] + 
           magic_data['swap'] + 
           magic_data['commission']
         ).cumsum()
         
         strategy_name = f"{magic_data['symbol'].iloc[-1]} ({magic})"
         
         # Add to strategy data list
         for _, row in magic_data.iterrows():
           strategy_data.append({
               'time': row['time'],
               'profit': row['cumulative_profit'],
               'magic': magic,
               'strategy': strategy_name
           })
      
      # Create final dataframe
      strat_df = pd.DataFrame(strategy_data)
      
      # Create the chart using strategy names
      if not strat_df.empty:
         fig = px.line(strat_df, x='time', y='profit', color='strategy',
                      title='Cumulative Profit by Strategy')
         
         fig.update_layout(xaxis_title='Time',
                          yaxis_title='Cumulative Profit',
                          template='ggplot2',
                          height=800,
                          legend_title_text='Symbol (Magic)')
         
         st.plotly_chart(fig, config={'responsive': True})

   #---------------------------------------------------------------#
   # Get strategy data in table
   if strat_df.empty:
       st.error("No deals data available for metrics")
   else:
      # Group by magic number to calculate strategy metrics
      strategy_metrics = []
      for magic in strat_df['magic'].unique():
         strategy_data = df_charts_filtered[df_charts_filtered['magic'] == magic]
         
         total_profit = strategy_data['profit'].sum() + strategy_data['swap'].sum() + strategy_data['commission'].sum()
         trades = int(len(strategy_data)/2)
         
         winning_trades = strategy_data[strategy_data['profit'] > 0]['profit']
         losing_trades = strategy_data[strategy_data['profit'] < 0]['profit']
         
         avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
         avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
         win_rate = len(winning_trades) / trades if trades > 0 else 0
         
         profit_factor = winning_trades.sum() / abs(losing_trades.sum()) if losing_trades.sum() != 0 else float('inf')
         
         # Calculate max drawdown for this strategy
         strategy_cumsum = (strategy_data['profit'] + strategy_data['swap'] + strategy_data['commission']).cumsum()
         running_max = strategy_cumsum.expanding().max()
         drawdown = strategy_cumsum - running_max
         max_dd = drawdown.min()
         
         strategy_metrics.append({
            'Symbol (Magic)': strat_df['strategy'][strat_df['magic'] == magic].iloc[0],
            'Profit': f"{total_profit:.2f}",
            'Trades': trades,
            'Avg Win': f"{avg_win:.2f}",
            'Avg Loss': f"{avg_loss:.2f}",
            'Win Rate': f"{win_rate:.2%}",
            'Profit Factor': f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞",
            'Max DD': f"{max_dd:.2f}" 
         })
      
      metrics_df = pd.DataFrame(strategy_metrics)
      st.dataframe(metrics_df, width='stretch', hide_index=True)

   #---------------------------------------------------------------#
   # Histograms for strategy profits
   col1, col2 = st.columns(2)
   with col1:
      if 'strat_df' in locals() and not strat_df.empty:
         # Group by strategy and sum profits
         strategy_profits = strat_df.groupby('strategy')['profit'].last().reset_index()
         
         fig = px.bar(strategy_profits, x='strategy', y='profit',
                     title='Strategy', color='profit', text='profit')
         fig.update_traces(texttemplate='%{text:.2s}', textposition="outside")
         fig.update_layout(
            xaxis_title='Strategy',
            yaxis_title='Profit',
            template='plotly_dark',
            xaxis_tickangle=-45,
            showlegend=False
         )
         fig.update_coloraxes(showscale=False)
         fig.update_traces(textfont_size=12, textangle=0, cliponaxis=False)

         st.plotly_chart(fig, config={'responsive': True})

   with col2:
      if 'df_charts_filtered' in locals() and not df_charts_filtered.empty:
         # Create monthly profit data - filter from cached deals directly
         monthly_data = df_charts_filtered.copy()
         monthly_data['month'] = monthly_data['time'].dt.to_period('M')
         monthly_profits = monthly_data.groupby('month')[['profit', 'swap', 'commission']].sum()
         monthly_profits['total_profit'] = monthly_profits['profit'] + monthly_profits['swap'] + monthly_profits['commission']
         monthly_profits = monthly_profits.reset_index()
         monthly_profits['month_str'] = monthly_profits['month'].astype(str)
         fig = px.bar(monthly_profits, x='month_str', y='total_profit',
                     title='Monthly Return', hover_data=['total_profit', 'month_str'], 
                     text='total_profit', color='total_profit')
         fig.update_traces(texttemplate='%{text:.2s}', textposition="outside")
         fig.update_layout(
            xaxis_title='Month',
            yaxis_title='Profit',
            template='plotly_dark',
         )
         fig.update_coloraxes(showscale=False)
         fig.update_traces(textfont_size=12, textangle=0, cliponaxis=False)

         st.plotly_chart(fig, config={'responsive': True})
   
   col1, col2 = st.columns(2)
   with col1:
      if 'df_charts_filtered' in locals() and not df_charts_filtered.empty:
         # Create weekday profit data - filter from cached deals directly
         weekday_data = df_charts_filtered.copy()
         weekday_data['weekday'] = weekday_data['time'].dt.day_name()
         weekday_profits = weekday_data.groupby('weekday')[['profit', 'swap', 'commission']].sum()
         weekday_profits['total_profit'] = weekday_profits['profit'] + weekday_profits['swap'] + weekday_profits['commission']
         weekday_profits = weekday_profits.reset_index()
         
         # Ensure correct order of weekdays
         weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
         weekday_profits['weekday'] = pd.Categorical(weekday_profits['weekday'], categories=weekday_order, ordered=True)
         weekday_profits = weekday_profits.sort_values('weekday')
         
         fig = px.bar(weekday_profits, x='weekday', y='total_profit',
                     title='Day of the Week', hover_data=['total_profit', 'weekday'])
         fig.update_layout(
            xaxis_title='Day of the Week',
            yaxis_title='Profit',
            template='plotly_dark'
         )
         st.plotly_chart(fig, config={'responsive': True})

   with col2:
      if 'df_charts_filtered' in locals() and not df_charts_filtered.empty:
         # Create buy vs sell profit data - filter from cached deals directly
         buy_sell_data = df_charts_filtered.copy()

         # Map deal types: 0 = Sell, 1 = Buy (these are exit deal types)
         type_mapping = {0: 'Sell', 1: 'Buy'}
         buy_sell_data['trade_type'] = buy_sell_data['type'].map(type_mapping)
         
         # Group by trade type and sum profits
         buy_sell_profits = buy_sell_data.groupby('trade_type')[['profit', 'swap', 'commission']].sum()
         buy_sell_profits['total_profit'] = buy_sell_profits['profit'] + buy_sell_profits['swap'] + buy_sell_profits['commission']
         buy_sell_profits = buy_sell_profits.reset_index()
         
         fig = px.bar(buy_sell_profits, x='trade_type', y='total_profit',
                     title='Buy vs Sell', 
                     hover_data=['total_profit', 'trade_type'],
                     color='total_profit')
         fig.update_layout(
            xaxis_title='Trade Type',
            yaxis_title='Profit',
            template='plotly_dark'
         )
         fig.update_coloraxes(showscale=False)
         st.plotly_chart(fig, config={'responsive': True})

#---------------------------------------------------------------#
# Heatmap for correlation                                       #
#---------------------------------------------------------------#

with heatmap:
   #---------------------------------------------------------------#
   # Get strategy correlation heatmap
   if not strat_df.empty:
      # Keep only the last occurrence of each time-strategy combination
      dedup_df = strat_df.drop_duplicates(subset=['time', 'strategy'], keep='last')
    
      pivot_df = dedup_df.pivot(index='time', columns='strategy', values='profit').fillna(0)
      corr_matrix = pivot_df.corr()

      fig = px.imshow(corr_matrix,
                     color_continuous_scale='RdBu_r',
                     aspect='auto',
                     title='Strategy Correlation Heatmap')
      
      # Add text annotations manually for correlation matrix
      fig.update_traces(
          text=corr_matrix.values,
          texttemplate='%{text:.2f}',
          textfont={"size": 10}
      )
      
      fig.update_layout(
          template='plotly_dark',
          height=800,
          xaxis_title='Strategy',
          yaxis_title='Strategy'
      )
      st.plotly_chart(fig, config={'responsive': True})

#---------------------------------------------------------------#
# Colour Profit Formatting                                      #
#---------------------------------------------------------------#

# Apply conditional formatting for profit/loss
def color_profit(val):
   """
   Apply red text color for negative values, green for positive
   """
   try:
      if pd.isna(val) or val == 0:
         return ''
      elif val > 0:
          return 'color: rgba(30, 144, 255, 1)'
      else:
           return 'color: rgba(180, 50, 40, 1)'
   except:
        return ''
   
#---------------------------------------------------------------#
# Trade History                                                 #
#---------------------------------------------------------------#

with history:
   # Use cached deals data with account-specific cache
   account_info = mt5.account_info()
   current_login = account_info.login if account_info else 0
   cached_deals = get_cached_deals_data(current_login)
   strategy_ids = {}
   df_trades = pd.DataFrame(columns=['time', 'EA', 'symbol', 'magic', 'type', 'volume', 'price', 'profit', 'swap', 'commission', 'comment'])

   # Create mapping of magic numbers to strategy comments for entry deals
   if not cached_deals.empty:
      # Create mapping of magic numbers to strategy comments for entry deals
      entry_deals = cached_deals[(cached_deals['entry'] == 0)]
      strategy_ids = dict(zip(entry_deals['magic'], entry_deals['comment'].str[:2]))

      # Filter for exit deals only (entry == 1) and exclude magic == 0
      df_history_filtered = cached_deals[(cached_deals['entry'] == 1)].copy()

      if not df_history_filtered.empty:
         # Process data (time is already converted in cached data)
         df_history_filtered = df_history_filtered.sort_values(by='time', ascending=False)

         # Map deal types: 0 = Buy, 1 = Sell (exit deals)
         df_history_filtered['type'] = df_history_filtered['type'].map({0: 'Sell', 1: 'Buy'})

         # Map strategy IDs to EA column
         df_history_filtered['EA'] = df_history_filtered['magic'].map(strategy_ids).fillna('-')
         
         # Select final columns and round numeric values
         df_trades = df_history_filtered[['time', 'symbol', 'magic', 'type', 'volume', 'profit', 'swap', 'commission', 'comment']].copy()

      else:
         st.info("No closed positions found")
   else:
      st.error("No deals data available")
      
   # Style the dataframe
   if not df_trades.empty:
      styled_df = df_trades.style.map(color_profit, subset=['profit']).format({
         'volume': '{:.2f}',
         'profit': '{:.2f}',
         'swap': '{:.2f}',
         'commission': '{:.2f}'
      })

      st.dataframe(styled_df, width='stretch', hide_index=True, column_config={
         "time": "Close Time",
         "symbol": "Symbol", 
         "magic": "Magic",
         "type": "Type",
         "volume": "Volume",
         "profit": "Profit",
         "swap": "Swap",
         "commission": "Commission",
         "comment": "Comment"
      })
   else:
      st.dataframe(df_trades, width='stretch', hide_index=True)



#==========================================================================================================================================#

