# Changes made to the streamlit_app.py

if use_live:
    # Updated Period selectbox options
    period = st.selectbox("Period", ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"])

    # Updated Interval selectbox for swing trading
    interval_ui = st.selectbox("Interval", ["1d", "1w", "1mo"], index=0)
    interval = "1wk" if interval_ui == "1w" else interval_ui
else:
    # Other existing code here unchanged
