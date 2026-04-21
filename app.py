import streamlit as st
import pandas as pd
import joblib 
import plotly.express as px
from google_play_scraper import Sort, reviews
from datetime import datetime

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Ojol Insights Pro v4.0", page_icon="🚖", layout="wide")

# --- CUSTOM CSS (Styling Dashboard & Perbaikan Warna Teks) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }

    /* Perbaikan Visual Selectbox di Sidebar agar Teks Terlihat */
    div[data-baseweb="select"] * {
        color: #1e293b !important; 
    }
    [data-testid="stWidgetLabel"] p {
        color: white !important;
        font-weight: 600;
    }

    /* Kartu Metrik Glassmorphism */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 20px !important;
        padding: 25px !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07) !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] { background-color: #1e293b !important; }
    [data-testid="stSidebar"] .stMarkdown p { color: white !important; }

    /* Tombol Utama */
    div.stButton > button {
        width: 100%;
        border-radius: 15px !important;
        height: 3.5em !important;
        background: linear-gradient(45deg, #0068c9, #00d4ff) !important;
        color: white !important;
        font-weight: 800 !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(0, 104, 201, 0.4) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_all_models():
    path = 'models/'
    try:
        m_s = joblib.load(path + 'model_sentimen.pkl')
        t_s = joblib.load(path + 'tfidf_sentimen.pkl')
        m_k = joblib.load(path + 'model_kategori.pkl')
        t_k = joblib.load(path + 'tfidf_kategori.pkl')
        return m_s, t_s, m_k, t_k
    except:
        return None, None, None, None

m_s, t_s, m_k, t_k = load_all_models()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3233/3233481.png", width=100)
    st.title("Settings")
    
    dict_apps = {
        "Grab": "com.grabtaxi.passenger",
        "Gojek": "com.gojek.app",
        "Maxim": "com.taxsee.taxsee",
        "inDrive": "sinet.startup.inDriver"
    }
    selected_app_name = st.selectbox("Pilih Aplikasi:", list(dict_apps.keys()))
    app_id = dict_apps[selected_app_name]
    selected_year = st.selectbox("Pilih Tahun Target:", list(range(2026, 2019, -1)), index=2)
    
    st.divider()
    run_btn = st.button("🚀 Jalankan Analisis")

# --- UI UTAMA ---
st.title("🚖 Dashboard Analisis Ojol (Mode Full Year)")
st.caption(f"Analisis data ulasan **{selected_app_name}** tahun **{selected_year}**")

if run_btn:
    all_year_data = []
    continuation_token = None
    status_msg = st.empty()

    with st.spinner(f"Mengambil ulasan..."):
        stop_scraping = False
        while not stop_scraping:
            try:
                res, continuation_token = reviews(
                    app_id, lang='id', country='id', sort=Sort.NEWEST, 
                    count=1000, continuation_token=continuation_token
                )
                if not res: break
                
                df_batch = pd.DataFrame(res)
                df_batch['at'] = pd.to_datetime(df_batch['at'])
                target_reviews = df_batch[df_batch['at'].dt.year == selected_year]
                
                if not target_reviews.empty:
                    all_year_data.extend(target_reviews.to_dict('records'))
                
                min_date = df_batch['at'].min()
                status_msg.info(f"📅 Memeriksa ulasan tanggal: **{min_date.strftime('%d %b %Y')}**")

                if min_date.year < selected_year or not continuation_token:
                    stop_scraping = True
            except Exception as e:
                st.error(f"Error Scraping: {e}")
                break

    if all_year_data:
        df_final = pd.DataFrame(all_year_data).drop_duplicates(subset=['content'])
        
        # PREDIKSI
        if m_s:
            texts = df_final['content'].fillna("")
            df_final['sentimen'] = m_s.predict(t_s.transform(texts))
            df_final['kategori'] = m_k.predict(t_k.transform(texts))

        # --- KPI METRICS ---
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Ulasan", f"{len(df_final):,}")
        
        if 'sentimen' in df_final.columns:
            
            top_kat = df_final['kategori'].value_counts().idxmax()
            m3.metric("Isu Utama", top_kat.upper())
            
            avg_star = df_final['score'].mean() if 'score' in df_final.columns else 0
            m4.metric("Avg. Rating", f"⭐ {avg_star:.1f}")

        # --- VISUALISASI ---
        st.markdown("### 📈 Analisis Visual")
        c1, c2 = st.columns([4, 6])
        
        with c1:
            st.markdown("**Distribusi Sentimen**")
            fig_pie = px.pie(df_final, names='sentimen', hole=0.6,
                           color_discrete_sequence=px.colors.sequential.Tealgrn_r)
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.markdown("**Top Isu Berdasarkan Kategori**")
            df_kat = df_final['kategori'].value_counts().reset_index()
            # Fix ValueError: Rename kolom secara eksplisit
            df_kat.columns = ['kategori', 'jumlah']
            
            fig_bar = px.bar(df_kat, x='kategori', y='jumlah', color='kategori',
                           color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig_bar, use_container_width=True)

            # Tambahan ringkasan otomatis
            st.markdown("**💡 Ringkasan Isu:**")
            for _, row in df_kat.iterrows():
                pct = (row['jumlah'] / len(df_final)) * 100
                st.write(f"- **{pct:.1f}%** ulasan mengomentari tentang **{row['kategori'].upper()}**")

        # --- TREND ---
        st.markdown("**Tren Ulasan Bulanan**")
        df_final['bulan'] = df_final['at'].dt.strftime('%b')
        order_m = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        trend = df_final.groupby(['bulan', 'sentimen']).size().reset_index(name='jumlah')
        trend['bulan'] = pd.Categorical(trend['bulan'], categories=order_m, ordered=True)
        trend = trend.sort_values('bulan')

        fig_trend = px.line(trend, x='bulan', y='jumlah', color='sentimen', markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)

        # --- DATA TABLE ---
        with st.expander("🔍 Detail Data"):
            st.dataframe(df_final[['at', 'userName', 'content', 'sentimen', 'kategori']], use_container_width=True)
    else:
        st.warning("Data tidak ditemukan.")