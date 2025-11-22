# ----------------------------------------
# This section imports all Python libraries used in the project.
# Includes data processing (pandas, numpy), visualization (plotly),
# UI rendering (streamlit), and optional machine learning tools.

import streamlit as st             # Streamlit arayüz kütüphanesi
import pandas as pd                # Veri işleme için pandas
import numpy as np                 # Sayısal hesaplamalar için numpy
import plotly.express as px        # Plotly'nin hızlı görselleştirme modülü
import plotly.graph_objects as go  # Daha karmaşık grafikler için Plotly GO
import os                          # Dosya işlemleri için OS modülü


# ----------------------------------------
# Machine learning tools like KMeans are optional.
# try/except ensures Streamlit won't crash if sklearn is not installed.

try:
    from sklearn.preprocessing import StandardScaler  # Verileri ölçeklemek için
    from sklearn.cluster import KMeans                # KMeans kümeleme algoritması
    SKLEARN_AVAILABLE = True                          # Sklearn mevcut bayrağı
except Exception:
    SKLEARN_AVAILABLE = False                         # Eğer import başarısızsa False


# ----------------------------------------
# Sets the dashboard title, icon, and wide layout.
# Standard initialization block for Streamlit applications.

st.set_page_config(
    page_title="Shopping Behavior Dashboard",            # Tarayıcı sekmesi başlığı
    page_icon="✈️",                                     # Sekme ikonu (emoji)
    layout="wide"                                        # Geniş ekran düzeni
)
 

# ----------------------------------------
# We added CSS to improve the look.
# Triple quotes (""") are used to define multi-line CSS strings.

st.markdown("""
<style>
h1 {
     font-size: 60px !important;               /* Başlık font büyüklüğünü belirtip !important ile override edilmesini önlüyoruz */         
     background: linear-gradient(90deg, #FF4B4B, #FF914D, #FFD700, #4CAF50, #1E90FF, #8A2BE2);
                                               /* Başlığa soldan sağa geçişli renk gradienti uyguluyoruz
                                                Renk sırası: kırmızı → turuncu → sarı → yeşil → mavi → mor */
            
     -webkit-background-clip: text;            /* Gradient’in sadece metin üzerinde görünmesini sağlar */          
     -webkit-text-fill-color: transparent;     /* Arka plan yerine gradientin gözükmesi için metni şeffaf yapar */    
     font-weight: bold;                        /* Kalın font uygular */  
     padding-bottom: 10px;                     /* Başlığın altına boşluk ekler, diğer içerik ile çakışmayı önler */  
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------
st.title("Shopping Behavior Dashboard")       # Sayfa başlığı
st.markdown("Welcome to the interactive dashboard for analyzing customer shopping trends.")
st.markdown("---")                            # Araya çizgi çekelim


# ----------------------------------------
# US State Codes Map

STATE_NAME_TO_ABBR = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO','Connecticut':'CT',
    'Delaware':'DE','Florida':'FL','Georgia':'GA','Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA',
    'Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI',
    'Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH',
    'New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND','Ohio':'OH','Oklahoma':'OK',
    'Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD','Tennessee':'TN',
    'Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA','Washington':'WA','West Virginia':'WV','Wisconsin':'WI',
    'Wyoming':'WY','District of Columbia':'DC'
}


# ----------------------------------------
# This function attempts to load the CSV file from a given path.
# If the file is missing or unreadable, it returns None instead of crashing.
# Streamlit's cache avoids re-reading the file on every interaction.

@st.cache_data
def load_data(filepath):          # CSV dosyasını yükler, hata olursa None döner
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        return None
    except Exception:
        return None

data_path = "Shopping_behavior.csv"
df = load_data(data_path)         # Veriyi yüklüyoruz

# Eğer bulunamazsa kullanıcıdan yüklemesini istiyoruz
if df is None:
    st.sidebar.warning(f"The dataset could not be found at:`{data_path}'")

    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully")
    else:
        st.info("Upload the dataset to continue")
        st.stop()


# ----------------------------------------
# Removes unnecessary columns, normalizes column names, converts numeric fields, 
# handles missing values and prepares categorical + location data for analysis.

cols_to_drop = [                     # Kullanılmayan sütunları kaldır
    "Review Rating",
    "Subscription Status",
    "Discount Applied",
    "Promo Code Used",
    "Previous Purchases"
]

for col in cols_to_drop:
    if col in df.columns:
        df.drop(columns=col, inplace=True)   # Varsa sutünu kaldır

df.columns = [c.strip() for c in df.columns]        # Sütun adlarını düzenle (boşluk vs varsa kaldır)

numeric_cols = ["Age", "Purchase Amount (USD)", "Frequency of Purchases"]   # Seçili alanları sayısal değere çevir
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

categorical_cols = [                                             # Kategorik sütunlardaki eksik değerleri doldur
    "Gender", "Category", "Season", "Payment Method", 
    "Shipping Type", "Size", "Color", "Item Purchased", "Location"
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown").astype(str)

for col in ["Age", "Purchase Amount (USD)"]:          # Eksik sayısal değeri olan satırları sil
    if col in df.columns:
        df = df[~df[col].isna()]

if "Location" in df.columns:                        # Coğrafi grafikler için eyalet isimlerini kodla  
    df["Location_Abbr"] = df["Location"].apply(
        lambda x: STATE_NAME_TO_ABBR.get(x, x) if pd.notna(x) else np.nan
    )
else:
    df["Location_Abbr"] = np.nan


# ----------------------------------------
# This section provides interactive controls for filtering the dataset.
# It allows users to customize the dashboard based on their analysis needs.

st.sidebar.header("🔎 Filters / Controls")     # Sidebar başlığı

season_list = sorted(df["Season"].dropna().unique().tolist()) if "Season" in df.columns else []  # Mevcut sezonları listele
season_sel = st.sidebar.selectbox(
    "Select Season",                                # Selectbox başlığı
    options=["All Seasons"] + season_list,          # 'Tümü' seçeneği ekle
    help="Filter data by season to see seasonal trends in shopping behavior."  # Tooltip açıklaması
)

cat_options = sorted(df["Category"].dropna().unique().tolist()) if "Category" in df.columns else []  # Kategori listesi
cat_sel = st.sidebar.multiselect(
    "Select Category (optional)",    # Kullanıcıya görünen başlık
    options=cat_options,             # Tüm kategoriler
    default=cat_options,             # Varsayılan: Hepsi seçili
    help="Choose one or more product categories to focus on specific items."    # Tooltip açıklaması
)

age_min = int(df["Age"].min()) if "Age" in df.columns else 18   # Minimum yaş
age_max = int(df["Age"].max()) if "Age" in df.columns else 90   # Maksimum yaş
age_range = st.sidebar.slider(
    "Age Range",            # Slider başlığı
    min_value=age_min,      # Slider alt sınır
    max_value=age_max,      # Slider üst sınır
    value=(age_min, age_max),   # Varsayılan başlangıç ve bitiş
    help="Filter customers by age to see trends for different age groups."   # Tooltip açıklaması
)

gender_options = sorted(df["Gender"].unique().tolist()) if "Gender" in df.columns else []  # Cinsiyet listesi
gender_sel = st.sidebar.multiselect(
    "Select Gender (optional)",   # Kullanıcı başlığı
    options=gender_options,       # Mevcut cinsiyetler
    default=gender_options,       # Varsayılan: Hepsi seçilir
    help="Filter by gender to compare shopping behavior of different groups."   # Tooltip açıklaması
)

pmin = int(df["Purchase Amount (USD)"].min()) if "Purchase Amount (USD)" in df.columns else 0   # Minimum harcama
pmax = int(df["Purchase Amount (USD)"].max()) if "Purchase Amount (USD)" in df.columns else 1000 # Maksimum harcama
price_range = st.sidebar.slider(
    "Purchase Amount Range (USD)",    # Başlık
    min_value=pmin,                    # Alt limit
    max_value=pmax,                    # Üst limit
    value=(pmin, pmax),                # Varsayılan değerler
    help="Filter by purchase amount to focus on low, medium, or high spenders."    # Tooltip açıklaması
)


# ---------------------------------------
# This section applies all user-selected filters to the dataset.
# Our purpose is that we should be ensure that only relevant records appear in visualizations.

filtered = df.copy()  # Orijinal verinin kopyasını oluşturur

if season_sel != "All Seasons":                                # Eğer tüm sezonlar seçilmediyse
    filtered = filtered[filtered["Season"] == season_sel]      # Veri sezona göre filtrelenir

if cat_sel:                                                    # Eğer en az 1 kategori seçilmişse
    filtered = filtered[filtered["Category"].isin(cat_sel)]    # Kategoriye göre filtreleme

filtered = filtered[
    (filtered["Age"] >= age_range[0]) &                        # Minimum yaş kontrolü
    (filtered["Age"] <= age_range[1])                          # Maksimum yaş kontrolü
]

filtered = filtered[filtered["Gender"].isin(gender_sel)]       # Cinsiyet filtrelemesi

filtered = filtered[
    (filtered["Purchase Amount (USD)"] >= price_range[0]) &    # Minimum harcama
    (filtered["Purchase Amount (USD)"] <= price_range[1])      # Maksimum harcama
]

st.sidebar.markdown(f"**Filtered Records:** {len(filtered)}")  # Kaç kayıt kaldığını gösterir

st.header("Shopping Insights Overview")  # Görselleştirmelerin ana başlığı


# ----------------------------------------
# 1) TREEMAP — TOTAL SPENDING BY CATEGORY
# Treemap chart showing total spending aggregated by product category.

st.subheader("Treemap")    # Grafik alt başlığı
st.markdown("This treemap shows the total purchase amount aggregated by product category, \
             giving a clear view of which categories contribute most to revenue.")

if "Category" in filtered.columns:
    tdf = filtered.groupby("Category")["Purchase Amount (USD)"].sum().reset_index()
    fig_treemap = px.treemap(tdf, 
                path=["Category"], 
                values="Purchase Amount (USD)",
                title="Total Spending by Category"
                )
    st.plotly_chart(fig_treemap, use_container_width=True)     # Grafiği ekranda göster


# ----------------------------------------
# 2) SANKEY DIAGRAM — CUSTOMER FLOW
# Visualizes customer flow from Category to Payment Method to Shipping Type.

st.subheader("Sankey Diagram")   # Grafik başlığı
st.markdown(
    "This Sankey diagram visualizes how customers move from product categories \
     to their preferred payment methods and shipping types, giving insight into purchasing behavior patterns.")

if all(c in filtered.columns for c in ["Category", "Payment Method", "Shipping Type"]):     # Gerekli kolonları kontrol ediyoruz
    cats = filtered["Category"].astype(str).unique().tolist()
    pays = filtered["Payment Method"].astype(str).unique().tolist()
    ships = filtered["Shipping Type"].astype(str).unique().tolist()
    
    labels = cats + pays + ships         # Tüm etiketleri tek listede birleştir

    def idx(x):                          # Etiket ismine göre index bulmak için yardımcı fonksiyon
        return labels.index(x)

    src, dst, val = [], [], []           # Sankey bağlantıları için listeler

    # Category → Payment Method
    cat_pay = filtered.groupby(["Category", "Payment Method"]).size().reset_index(name="count")
    for _, r in cat_pay.iterrows():
        src.append(idx(r["Category"]))
        dst.append(idx(r["Payment Method"]))
        val.append(int(r["count"]))

    # Payment Method → Shipping Type
    pay_ship = filtered.groupby(["Payment Method", "Shipping Type"]).size().reset_index(name="count")
    for _, r in pay_ship.iterrows():
        src.append(idx(r["Payment Method"]))
        dst.append(idx(r["Shipping Type"]))
        val.append(int(r["count"]))

    # Bağlantı yoksa grafik gösterme
    if sum(val) > 0:
        sankey_fig = go.Figure(
          data=[ go.Sankey
                    ( node=dict(
                            label=labels,
                            pad=15,
                            thickness=15
                        ),
                    link=dict(
                            source=src,
                            target=dst,
                            value=val
                        )
                    )
                ]
            )
        
    sankey_fig.update_layout(
        title_text="Customer Flow: Category → Payment Method → Shipping Type",
        font_size=10
        )
    
    st.plotly_chart(sankey_fig, use_container_width=True)     # Grafiği ekranda göster
   
else:
    st.info("Not enough data to generate the Sankey diagram.")


# ----------------------------------------
# HISTOGRAM — AGE DISTRIBUTION
# Histogram showing the distribution of customer ages.

st.subheader("Age Distribution Histogram")   # Grafik başlığı
st.markdown(
    "This histogram displays the distribution of customer ages."
    "which age groups are more active shoppers. ")

if "Age" in filtered.columns:                # Age kolonu mevcut mu kontrol edilir

 fig_hist = px.histogram(                    # Histogram çizimi için Plotly kullanıyoruz
        filtered,
        x="Age",                             # Histogram ekseni
        nbins=20,                            # Histogram kutu sayısı
        title="Customer Age Distribution",
        marginal="box",                      # Üstte boxplot göster
        hover_data=["Gender", "Category"]    # Üzerine basınca ek bilgi 
 )
 st.plotly_chart(fig_hist, use_container_width=True)    

# ----------------------------------------
# SUNBURST CHART — SEASON → CATEGORY SALES
# Shows total spending grouped by Season and Category

st.subheader("Sunburst Chart")
st.markdown(
    "This sunburst chart visualizes how total purchase amounts are distributed "
    "across different seasons and product categories."
)

if not filtered.empty:
    # Create a sunburst chart showing hierarchy: Season -> Category
    fig_sunburst = px.sunburst(
        filtered,
        path=["Season", "Category"],
        values="Purchase Amount (USD)",
        color_continuous_scale="Viridis",
        color="Purchase Amount (USD)",
        title="Sales Distribution by Season and Category",
        hover_data={"Purchase Amount (USD)": ":,.2f"}
    )
    st.plotly_chart(fig_sunburst, use_container_width=True)
else:
    st.warning("No data available for selected filters.")



# ----------------------------------------
# BAR CHART — TOTAL SALES BY ITEM

st.subheader("Bar Chart")
st.markdown(
    "This bar chart displays the total purchase amount for each item, highlighting the top-selling products."
)

if not filtered.empty:
    # Aggregate total sales per item
    item_sales = (
        filtered.groupby("Item Purchased")["Purchase Amount (USD)"].sum().reset_index()
    )

    # Sort descending to show top-selling items at the top
    item_sales = item_sales.sort_values("Purchase Amount (USD)", ascending=False)

    # Create bar chart
    fig_bar = px.bar(
        item_sales,
        x="Item Purchased",
        y="Purchase Amount (USD)",
        color_continuous_scale="Plasma",
        color="Purchase Amount (USD)",
        title="Total Sales by Item",
        hover_data={"Item Purchased": False, "Purchase Amount (USD)": ":,.2f"}
    )

    st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.warning("No data available for the selected filters.")


# ----------------------------------------
# GEOGRAPHIC MAP — TOTAL PURCHASE AMOUNT BY STATE

st.subheader("Geographical Distribution of Total Spending")
st.markdown(
    "This choropleth map shows total purchase amounts across U.S. states. "
    
)

# Eğer Location_Abbr varsa işle
if "Location_Abbr" in filtered.columns and not filtered.empty:
    # Aggregate total sales per state
    geo_df = (
        filtered.groupby("Location_Abbr")["Purchase Amount (USD)"]
        .sum()
        .reset_index()
    )

    fig_geo = px.choropleth(
        geo_df,
        locations="Location_Abbr",          # State abbreviations
        locationmode="USA-states",
        color="Purchase Amount (USD)",      # Color by spending
        color_continuous_scale="Viridis",
        scope="usa",
        title="Total Purchase Amount by US State"
    )

    st.plotly_chart(fig_geo, use_container_width=True)

else:
    st.warning("Not enough geographical data to generate the map.")



# ----------------------------------------
# SCATTER PLOT — AGE vs PURCHASE AMOUNT
st.subheader("Scatter Plot: Age vs Purchase Amount")
st.markdown("This scatter plot visualizes the relationship between customer age and how much they spend.")

if all(c in filtered.columns for c in ["Age", "Purchase Amount (USD)"]):
    fig_scatter = px.scatter(
        filtered,
        x="Age",
        y="Purchase Amount (USD)",
        color="Gender",
        title="Age vs Purchase Amount",
        hover_data=["Category", "Season"]
    )
    st.plotly_chart(fig_scatter, use_container_width=True)



# ----------------------------------------
# HEATMAP — CORRELATION MATRIX
st.subheader("6. Heatmap: Location vs Category")

if len(filtered) > 0:
    # 1. Pivot the Data
    # We use state abbreviations to keep the axis cleaner
    loc_col = 'Location_Abbr' if 'Location_Abbr' in filtered.columns else 'Location'

    heatmap_data = filtered.pivot_table(
        index=loc_col,
        columns='Category',
        values='Purchase Amount (USD)',
        aggfunc='mean'
    ).fillna(0)

    # 2. Sorting - Makes the chart more readable
    # Sort rows (States) by average spending (Ascending sort works best for the Y-axis visual)
    heatmap_data['Total_Mean'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values('Total_Mean', ascending=True).drop(columns='Total_Mean')

    # 3. Visualization
    fig_heatmap = px.imshow(
        heatmap_data,
        title="Average Purchase Amount ($) - State vs Category",
        labels=dict(x="Category", y="State (Abbr)", color="Avg Amount"),
        color_continuous_scale='YlGnBu',  # A fresher color palette (Yellow-Green-Blue)
        text_auto=".0f",  # Display values inside cells (no decimals)
        aspect="auto"
    )

    # 4. Fine-tuning Layout
    fig_heatmap.update_layout(
        height=800,  # Increased height to prevent state labels from overlapping
        xaxis_title=None,  # Avoid unnecessary label clutter
        yaxis_title=None,
        xaxis=dict(side="top")  # Move categories to the top for better readability
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)
else:
    st.warning("No data available to display with these filters.")


# ----------------------------------------
# PARALLEL COORDINATES
st.subheader("Advanced Multi-Dimensional Analysis")
st.markdown("""
**This chart reveals complex relationships within the dataset.** By following the lines from left to right, you can visualize a complete customer journey: 
starting from their gender, moving through the season, product category, purchase amount, 
and finally their chosen payment method.
""")

# 1. Data Preparation: Encoding categorical data into numbers
# This process allows text values like "Winter" or "Credit Card" to appear on the axes.

plot_df = filtered.copy()

# Define categorical columns and their display labels
cat_features = {
    "Gender": "Gender",
    "Season": "Season",
    "Category": "Category",
    "Payment Method": "Payment Method",
    "Review Rating": "Rating (1-5)"
}

# Numerical columns
num_features = ["Age", "Purchase Amount (USD)"]

# Create a list of dimensions (axes) for the plot
dimensions = []

# -- Gender Axis --
if "Gender" in plot_df.columns:
    # Convert categories to codes (e.g., Female=0, Male=1)
    plot_df["Gender_Code"] = plot_df["Gender"].astype("category").cat.codes
    # Get the actual names for the axis ticks
    g_names = plot_df["Gender"].astype("category").cat.categories

    dimensions.append(dict(
        range=[0, len(g_names) - 1],
        tickvals=list(range(len(g_names))),
        ticktext=list(g_names),
        label="Gender",
        values=plot_df["Gender_Code"]
    ))

# -- Age Axis (Numerical) --
if "Age" in plot_df.columns:
    dimensions.append(dict(
        range=[plot_df["Age"].min(), plot_df["Age"].max()],
        label="Age",
        values=plot_df["Age"]
    ))

# -- Season Axis --
if "Season" in plot_df.columns:
    plot_df["Season_Code"] = plot_df["Season"].astype("category").cat.codes
    s_names = plot_df["Season"].astype("category").cat.categories

    dimensions.append(dict(
        range=[0, len(s_names) - 1],
        tickvals=list(range(len(s_names))),
        ticktext=list(s_names),
        label="Season",
        values=plot_df["Season_Code"]
    ))

# -- Category Axis --
if "Category" in plot_df.columns:
    plot_df["Cat_Code"] = plot_df["Category"].astype("category").cat.codes
    c_names = plot_df["Category"].astype("category").cat.categories

    dimensions.append(dict(
        range=[0, len(c_names) - 1],
        tickvals=list(range(len(c_names))),
        ticktext=list(c_names),
        label="Category",
        values=plot_df["Cat_Code"]
    ))

# -- Purchase Amount Axis (Numerical - Main Focus) --
if "Purchase Amount (USD)" in plot_df.columns:
    dimensions.append(dict(
        range=[plot_df["Purchase Amount (USD)"].min(), plot_df["Purchase Amount (USD)"].max()],
        label="Spend ($)",
        values=plot_df["Purchase Amount (USD)"]
    ))

# -- Payment Method Axis --
if "Payment Method" in plot_df.columns:
    plot_df["Pay_Code"] = plot_df["Payment Method"].astype("category").cat.codes
    p_names = plot_df["Payment Method"].astype("category").cat.categories

    dimensions.append(dict(
        range=[0, len(p_names) - 1],
        tickvals=list(range(len(p_names))),
        ticktext=list(p_names),
        label="Payment",
        values=plot_df["Pay_Code"]
    ))

# -- Review Rating Axis --
if "Review Rating" in plot_df.columns:
    dimensions.append(dict(
        range=[1, 5],
        label="Rating",
        values=plot_df["Review Rating"]
    ))

# 2. Generating the Chart (Using Go.Parcoords)
if len(plot_df) > 0:
    fig_complex = go.Figure(data=
    go.Parcoords(
        line=dict(
            color=plot_df["Purchase Amount (USD)"],  # Color lines based on spending
            colorscale="Turbo",  # Vibrant, high-contrast color scale
            showscale=True,
            cmin=plot_df["Purchase Amount (USD)"].min(),
            cmax=plot_df["Purchase Amount (USD)"].max()
        ),
        dimensions=dimensions  # The list of axes we prepared above
    )
    )

    fig_complex.update_layout(
        title="Complex Analysis: Customer Journey Flow",
        height=600,  # Increase height for better visibility
        font=dict(size=11)  # Adjust font size for readability
    )

    st.plotly_chart(fig_complex, use_container_width=True)
else:
    st.warning("Not enough data available to generate this chart.")


# ----------------------------------------
# DATA PREVIEW & CSV DOWNLOAD
# Section for previewing filtered data and downloading it as CSV.

st.markdown("---")                                   # Bölüm ayırıcı çizgi 
st.subheader("Filtered Data Preview & Download")     # Alt baslık

st.markdown(                                         # Kullanıcı bilgilendirmesi  
    "View a snapshot of the filtered dataset below. You can also download the \
    filtered data as a CSV file for further analysis or reporting."
)
st.dataframe(filtered.head(50))                      # Datasetin ilk 50 satırını tablo olarak göster

def df_to_csv_bytes(df_):
    return df_.to_csv(index=False).encode("utf-8") 

csv_data = df_to_csv_bytes(filtered)                # İndirilebilir formata getir
 
st.download_button(
    label="Download Filtered Data (CSV)",    
    data=csv_data,                         
    file_name="filtered_shopping_behavior.csv",  
    mime="text/csv"                            
)