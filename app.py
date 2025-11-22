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