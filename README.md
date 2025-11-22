# Shopping Behavior Dashboard  
**Interactive Data Visualization App built with Streamlit & Plotly**

This project is an interactive dashboard designed to analyze customer shopping patterns using advanced visualizations, filters, and geographical insights.  
Users can explore spending behavior, seasonal trends, product categories, payment methods and more.

---

## Features

###  **1. Interactive Filters**
- Season filter  
- Category selection  
- Age range slider  
- Gender selection  
- Purchase amount range  
- Automatic dataset cleaning  

These filters dynamically update every visualization on the dashboard.

---

## **2. Visualizations Included**

### **Treemap**  
Treemap chart showing total spending aggregated by product category.

### **Sankey Diagram**  
This Sankey diagram visualizes how customers move from product categories to their preferred payment methods
and shipping types, giving insight into purchasing behavior patterns
For example; this project displays the flow from:
`Category → Payment Method → Shipping Type`

### **Histogram**  
Shows the distribution of customer ages.

### **Sunburst Chart**  
Shows total spending grouped by Season and Category

### **Bar Chart**  
Displays the total purchase amount for each item, highlighting the top-selling products.

### **Geographic Map**  
This choropleth map shows total purchase amounts across U.S. states.
    
### **Scatter Plot**  
Visualizes the relationship between customer age and how much they spend.

### **Heatmap**  
Shows average spending for each category by state.

### **Parallel Coordinates**  
Advanced multi-dimensional analysis combining:
- Gender  
- Age  
- Season  
- Category  
- Spend  
- Payment Method  

---

## Dataset  
The dashboard uses:
If the dataset is missing, users can upload it through the sidebar.

---

## Data Cleaning & Preparation
The application automatically:
- Removes unused columns  
- Converts numeric fields  
- Fills missing categorical values  
- Maps US state names → abbreviations  

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Base language |
| **Streamlit** | Web dashboard framework |
| **Pandas / NumPy** | Data transformation |
| **Plotly Express & Graph Objects** | Interactive visualizations |

---

## How to Run the App

### **1. Clone the Repository**
```bash
cd DataVis
git clone https://github.com/mratdrn/DataVis.git

2. Install Dependencies
pip install streamlit pandas numpy plotly

3. Run the Streamlit App
streamlit run app.py

Download
Users can download the filtered dataset as CSV directly from the dashboard.

Project Structure
DataVis/
│── app.py                # Main Streamlit application
│── Shopping_behavior.csv # Dataset (optional)
│── README.md             # Project documentation


Contributors:
- Murat Durna
- Melike Kara
- Almina Pala

---
