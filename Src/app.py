import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from model_metrics import get_model_accuracy

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_file(file_name):
    """Helper function to load files with proper path resolution"""
    file_path = os.path.join(SCRIPT_DIR, file_name)
    
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{file_path} does not exist. Please check the path.")
    
    if file_name.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        return joblib.load(file_path)

# Load all files with proper path handling
try:
    # Load data
    data2 = load_file('processed_data.csv')
    
    # Load model and encoder
    model = load_file('model.pkl')
    le_item = load_file('label_encoder.pkl')
    
    # Get model accuracy
    accuracy = get_model_accuracy()

except FileNotFoundError as e:
    st.error(f"Failed to load required files: {e}")
    st.error("Please ensure these files are in your GitHub repository:")
    st.error("- processed_data.csv")
    st.error("- model.pkl") 
    st.error("- label_encoder.pkl")
    st.stop()

# Rest of your app code...

# ... continue with your existing app code

# Rest of your app code...

# ... continue with your existing app code

st.title("Inventory Management System with ML Predictions")
#st.write(f"Model Accuracy: {accuracy:.2f}")

# Dashboard: Display Earliest and Latest Purchased Items
st.header("Latest Purchased Items")
sorted_items = data2.sort_values(by='date_of_purchased')
#earliest_purchased = sorted_items.head(5)
latest_purchased = sorted_items.tail(5)


#st.subheader("Latest Purchased Items")
st.dataframe(latest_purchased[['item_name', 'date_of_purchased', 'quantity']])

# Stockout Level for Consumable Items
st.header("Stockout Levels for Consumable Items")
low_stock_threshold = 3  # Define your threshold here
low_stock_items = data2[(data2['is_consumable'] == True) & (data2['quantity'] < low_stock_threshold)]

if not low_stock_items.empty:
    st.dataframe(low_stock_items[['item_name', 'quantity']])
else:
    st.success("All consumable items are sufficiently stocked.")

# Visualizations: Stocked-Out Items and Items with Most Quantity
# Stock Out Items (assuming quantity = 0 represents stocked out)
stocked_out_items = data2[data2['quantity'] ==0]
if not stocked_out_items.empty:
    st.subheader("Stocked Out Items")
    st.bar_chart(stocked_out_items.set_index('item_name')['quantity'])
else:
    st.subheader("Stocked Out Items")
    st.write("No stocked out items.")
# Create a bar chart for stocked out items

# Minimum stock 
min_stock_items = data2[data2['quantity'] <= 2]

# Create a bar chart for stocked out items
if not min_stock_items.empty:
    st.subheader("Low stock level")
    st.bar_chart(min_stock_items.set_index('item_name')['quantity'])
else:
    st.write("No stocked out items.")

# Items with the Most Quantity
most_quantity_items = data2.nlargest(10, 'quantity')  # Get top 10 items by quantity
st.subheader("Items with highest stock")
st.bar_chart(most_quantity_items.set_index('item_name')['quantity'])

# Predicting which project might need which item
st.header("Project Item Prediction")
selected_item = st.selectbox("Select an Item", data2['item_name'].unique())
if selected_item:
    selected_item_encoded = le_item.transform([selected_item])[0]
    predicted_project = model.predict([[selected_item_encoded, 10]])  # Assuming quantity of 10
    st.write(f"The projects that might need '{selected_item}' is: {predicted_project} ")
