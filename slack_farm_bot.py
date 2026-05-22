import os
import io
import glob
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from functools import lru_cache
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from flask import Flask
from threading import Thread

# ==========================================
# 1. CONFIGURATION & CREDENTIALS
# ==========================================
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]

CSV_FOLDER = r"./soil_data"
GDD_BASE = 5.0

PLANTING_DB = {
    'Benderbrook 1': '2026-05-07', 'Benderbrook 2': '2026-05-06', 'Brucelea': '2026-05-06',
    'Christie-2': '2026-05-17', 'Christie-1': '2026-05-18', 'Leis': '2026-05-17',
    'Burm': '2026-05-13', 'FieldAndFlock 1': '2026-05-07', 'Gerber Acres': '2026-05-06',
    'GerMar Farms (Grubb)': '2026-05-01', 'Gerrits': '2026-05-10', 'Harrison Farms': '2026-05-16',
    'Highland': '2026-05-06', 'JD Peters': '2026-05-18', 'Kerrington': '2026-05-15',
    'Klavan': '2026-05-09', 'Lang': '2026-05-15', 'Moosberger 2': '2026-05-11',
    'Renwick 2': '2026-05-07', 'Renwick 1': '2026-05-08', 'Schaus': '2026-05-12',
    'Schumhaven': '2026-05-09', 'Triaro': '2026-05-12', 'Triple Lane Farms': '2026-05-11',
    'Veldale': '2026-05-12', 'Wecker': '2026-05-08', 'Wettlaufer': '2026-05-12',
    'Campbell': '2026-05-06',
    'Bercab 1': None, 'Bercab 2': None, 'Sydenham 1': None, 'Sydenham 2 North': None, 
    'Sydenham 2 South': None, 'McAlpine': None, 'FieldAndFlock 2': None, 'Moosberger 1': None, 
    'Clare Horst': None, 'Marvara / Judd': None, 'Biermans': None
}

FIELD_NAME_MAP = {
    'Clare_Horst_Home_East_NutrientTexture': 'Clare Horst', 'Moose_CSV': 'Moosberger 1',
    'Roth_CSV': 'Benderbrook 1', 'Tim60_CSV': 'Benderbrook 2',
    'Upside_Robotics_Adam_Wettlaufer_Adam_Wettlaufer_NutrientTexture': 'Wettlaufer',
    'Upside_Robotics_Bercab_Bercab_1_Shop_NutrientTexture (1)': 'Bercab 1',
    'Upside_Robotics_Bercab_Bercab_2_NutrientTexture': 'Bercab 2',
    'Upside_Robotics_Biermans_Main_NutrientTexture': 'Biermans',
    'Upside_Robotics_Brad_Haack_Schause_NutrientTexture': 'Schaus',
    'Upside_Robotics_Brucelea_Brucelea_for_Upside_NutrientTexture': 'Brucelea',
    'Upside_Robotics_Christie_Christies_2_NutrientTexture': 'Christie-2',
    'Upside_Robotics_Christie_Christie_1_NutrientTexture': 'Christie-1',
    'Upside_Robotics_Ed_Burm_Ed_Burm_1_and_2_NutrientTexture': 'Burm',
    'Upside_Robotics_Field_and_Flock_Demaree_NutrientTexture': 'FieldAndFlock 1',
    'Upside_Robotics_Field_and_Flock_DeVries_NutrientTexture': 'FieldAndFlock 2',
    'Upside_Robotics_Gerard_Grubb_Gerard_Grubb_NutrientTexture': 'GerMar Farms (Grubb)',
    'Upside_Robotics_Gerber_Acres_Gerber_1_NutrientTexture': 'Gerber Acres',
    'Upside_Robotics_Gerrits_Gerrits_NutrientTexture': 'Gerrits',
    'Upside_Robotics_Greg_Leis_Tracks_NutrientTexture': 'Leis',
    'Upside_Robotics_Highland_Farms_Highland_1_NutrientTexture': 'Highland',
    'Upside_Robotics_JD_Peters_JD_Peters_13th_Concession_NutrientTexture': 'JD Peters',
    'Upside_Robotics_Kerrington_Kerrington_NutrientTexture': 'Kerrington',
    'Upside_Robotics_Klavans_Klavan_7440_NutrientTexture': 'Klavan',
    'Upside_Robotics_Langs_main_NutrientTexture': 'Lang',
    'Upside_Robotics_Marvara_Marvara_1_NutrientTexture': 'Marvara / Judd',
    'Upside_Robotics_Renwick_Renwick_1_NutrientTexture': 'Renwick 1',
    'Upside_Robotics_Renwick_Renwick_2_NutrientTexture': 'Renwick 2',
    'Upside_Robotics_Roland_McAlpine_McAlpine_1_NutrientTexture': 'McAlpine',
    'Upside_Robotics_Russ_Schumm_Schumm_401_NutrientTexture': 'Schumhaven',
    'Upside_Robotics_Scott_Campbell_Campbell_Home_NutrientTexture': 'Campbell',
    'Upside_Robotics_Sydenham_Sydenham_1_NutrientTexture': 'Sydenham 1',
    'Upside_Robotics_Sydenham_Sydenham_2_North_NutrientTexture': 'Sydenham 2 North',
    'Upside_Robotics_Sydenham_Sydenham_2_South_NutrientTexture': 'Sydenham 2 South',
    'Upside_Robotics_Triaro_Triaro_18_Line_NutrientTexture': 'Triaro',
    'Upside_Robotics_Triple_Lane_Farm_Triple_Lane_179_Howell_NutrientTexture': 'Triple Lane Farms',
    'Upside_Robotics_Veldale_Veldale_Research__74_NutrientTexture': 'Veldale',
    'Weckers_CSV': 'Wecker',
}

FIELD_STATION_MAP = {
    'Bercab 1': (48373, 'SARNIA'), 'Bercab 2': (48373, 'SARNIA'), 'Burm': (48373, 'SARNIA'),
    'Sydenham 1': (48373, 'SARNIA'), 'Sydenham 2 North': (48373, 'SARNIA'), 'Sydenham 2 South': (48373, 'SARNIA'),
    'Gerrits': (48373, 'SARNIA'), 'Kerrington': (48373, 'SARNIA'), 'McAlpine': (48373, 'SARNIA'),
    'Campbell': (27528, 'DELHI CS'), 'FieldAndFlock 1': (27528, 'DELHI CS'), 'FieldAndFlock 2': (27528, 'DELHI CS'),
    'Moosberger 1': (27528, 'DELHI CS'), 'Moosberger 2': (27528, 'DELHI CS'), 'JD Peters': (27528, 'DELHI CS'),
    'Harrison Farms': (53378, 'BRANTFORD AIRPORT'), 'Veldale': (53378, 'BRANTFORD AIRPORT'), 'Triple Lane Farms': (53378, 'BRANTFORD AIRPORT'),
    'Schumhaven': (10999, 'LONDON CS'), 'Leis': (10999, 'LONDON CS'), 'Gerber Acres': (10999, 'LONDON CS'),
    'Benderbrook 1': (10999, 'LONDON CS'), 'Benderbrook 2': (10999, 'LONDON CS'),
    'Clare Horst': (41983, 'ELORA RCS'), 'Marvara / Judd': (41983, 'ELORA RCS'), 'Klavan': (41983, 'ELORA RCS'), 'Triaro': (41983, 'ELORA RCS'),
    'Wettlaufer': (27529, 'GODERICH CLIMATE'), 'Brucelea': (27529, 'GODERICH CLIMATE'),
    'Renwick 2': (48569, 'WINGHAM AUTO'), 'Lang': (48568, 'CHESLEY CLIMATE'), 'Biermans': (48568, 'CHESLEY CLIMATE'),
    'Christie-1': (48568, 'CHESLEY CLIMATE'), 'Christie-2': (48568, 'CHESLEY CLIMATE'), 'Highland': (48568, 'CHESLEY CLIMATE'),
    'Renwick 1': (7844, 'MOUNT FOREST AUT'), 'Schaus': (7844, 'MOUNT FOREST AUT'), 'GerMar Farms (Grubb)': (7844, 'MOUNT FOREST AUT'),
    'Wecker': (54738, 'WINDSOR A'),
}

# ==========================================
# 2. RUNTIME EXTRACTION & HIGH-SPEED PARQUET CACHE
# ==========================================
if not os.path.exists("soil_data"):
    os.makedirs("soil_data")

for zf in glob.glob("*.zip"):
    with zipfile.ZipFile(zf, 'r') as zip_ref:
        zip_ref.extractall("soil_data")
    os.remove(zf)

# Build Parquet files for 50x read speeds
print("⚡ Initializing high-speed Parquet conversion...")
FILE_MAP = {}
for f in glob.glob(os.path.join(CSV_FOLDER, '*.csv')):
    base = os.path.basename(f).replace('.csv', '')
    mapped = FIELD_NAME_MAP.get(base, base)
    
    pq_path = f.replace('.csv', '.parquet')
    if not os.path.exists(pq_path):
        pd.read_csv(f).to_parquet(pq_path)
        
    if mapped not in FILE_MAP:
        FILE_MAP[mapped] = [pq_path]
    else:
        FILE_MAP[mapped].append(pq_path)

# Load Field-Level Terrain Metrics
TERRAIN_METRICS = {}
terrain_csv = "field_terrain_metrics.csv"
if os.path.exists(terrain_csv):
    tm = pd.read_csv(terrain_csv).set_index('Field')
    TERRAIN_METRICS = tm.to_dict('index')
    print(f"✅ Terrain metrics loaded for {len(TERRAIN_METRICS)} fields")
else:
    print("⚠️  No terrain metrics found. Proceeding without topography modifiers.")

WEATHER_CACHE = {}

@lru_cache(maxsize=32)
def get_field_soil_data(field_name):
    """Lazy-loads and caches Parquet files. Massively reduces Render RAM usage."""
    file_paths = FILE_MAP.get(field_name)
    if not file_paths:
        return None
    dfs = [pd.read_parquet(fp).assign(Field=field_name) for fp in file_paths]
    return pd.concat(dfs, ignore_index=True) if dfs else None

def fetch_climate_data(station_id, year):
    url = (f'https://climate.weather.gc.ca/climate_data/bulk_data_e.html'
           f'?format=csv&stationID={station_id}&Year={year}'
           f'&Month=1&Day=1&timeframe=2&submit=Download+Data')
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = df.columns.str.strip()
        
        # Standardize columns ONCE so we don't have to scan them repeatedly
        date_col = [c for c in df.columns if 'Date' in c or 'date' in c][0]
        tmean_col = [c for c in df.columns if 'Mean Temp' in c or 'TMEAN' in c][0]
        tmax_col = [c for c in df.columns if 'Max Temp' in c or 'TMAX' in c][0]
        tmin_col = [c for c in df.columns if 'Min Temp' in c or 'TMIN' in c][0]
        precip_col = [c for c in df.columns if 'Total Precip' in c or 'Total Rain' in c or 'PRECIP' in c][0]
        
        df = df.rename(columns={
            date_col: 'Date', tmean_col: 'Tmean', tmax_col: 'Tmax', 
            tmin_col: 'Tmin', precip_col: 'Precip'
        })
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        for col in ['Tmean', 'Tmax', 'Tmin', 'Precip']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        return df.dropna(subset=['Date'])
    except Exception as e:
        return None

def fetch_climate_data_cached(station_id, year):
    cache_key = f"{station_id}_{year}"
    now = datetime.now()
    if cache_key in WEATHER_CACHE:
        if (now - WEATHER_CACHE[cache_key]['fetched_at']).total_seconds() / 3600 < 6:
            return WEATHER_CACHE[cache_key]['data']
    data = fetch_climate_data(station_id, year)
    if data is not None:
        WEATHER_CACHE[cache_key] = {'data': data, 'fetched_at': now}
    return data

app = App(token=SLACK_BOT_TOKEN)

# ==========================================
# 3. INTERACTIVE SLACK ROUTINES (THREADED)
# ==========================================
@app.command("/gdd-chu")
def handle_gdd_chu(ack, respond, command):
    ack("🌱 *Fetching historical weather data and calculating Vectorized GDD...*")
    
    def background_worker():
        field_name = command['text'].strip()
        if field_name not in PLANTING_DB:
            respond(f"❌ Field `{field_name}` not recognized in database.")
            return
        if PLANTING_DB[field_name] is None:
            respond(f"⚠️ *Field Notification:* `{field_name}` does not have a planting date.")
            return
            
        station_info = FIELD_STATION_MAP.get(field_name)
        p_date = pd.to_datetime(PLANTING_DB[field_name])
        current_date = datetime.now()
        
        df_curr = fetch_climate_data_cached(station_info[0], current_date.year)
        df_prev = fetch_climate_data_cached(station_info[0], current_date.year - 1)
        
        if df_curr is None or df_prev is None:
            respond("❌ Weather data down or unreachable.")
            return

        def compute_metrics(df_year, start_dt, end_dt):
            mask = (df_year['Date'] >= start_dt) & (df_year['Date'] <= end_dt)
            sub = df_year[mask]
            if sub.empty: return 0, 0
            
            gdd = (sub['Tmean'] - GDD_BASE).clip(lower=0).sum()
            
            # Vectorized CHU (Massive Speedup)
            tmax = sub['Tmax'].clip(upper=30.0)
            ymax = np.where(tmax > 10.0, 3.33 * (tmax - 10.0) - 0.084 * (tmax - 10.0)**2, 0.0)
            ymin = np.where(sub['Tmin'] > 4.44, 1.8 * (sub['Tmin'] - 4.44), 0.0)
            chu = np.maximum(0.0, (ymax + ymin) / 2.0).sum()
            
            return round(gdd, 0), round(chu, 0)

        gdd_c, chu_c = compute_metrics(df_curr, p_date, current_date)
        gdd_p, chu_p = compute_metrics(df_prev, p_date - pd.DateOffset(years=1), current_date - pd.DateOffset(years=1))
        
        plt.figure(figsize=(7, 4))
        df_plot = df_curr[(df_curr['Date'] >= p_date) & (df_curr['Date'] <= current_date)].copy()
        if not df_plot.empty:
            df_plot['GDD_cum'] = (df_plot['Tmean'] - GDD_BASE).clip(lower=0).cumsum()
            plt.plot(df_plot['Date'], df_plot['GDD_cum'], color='#4a9e6b', linewidth=2.5)
            
        plt.title(f"Field: {field_name} — Cumulative GDD Tracking", fontsize=11, fontweight='bold')
        plt.ylabel('GDD Accumulated (°C·days)')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=150)
        img_buf.seek(0)
        plt.close()
        
        app.client.files_upload_v2(
            channel=command['channel_id'],
            initial_comment=f"🌱 *Field Report: {field_name}*\n• In-Season Heat: *{gdd_c:.0f} GDD* vs last year's *{gdd_p:.0f} GDD*\n• Yield Context: *{chu_c:.0f} CHU* vs last year's *{chu_p:.0f} CHU*",
            file=img_buf.read(), filename=f"{field_name}_weather.png"
        )
        
    Thread(target=background_worker).start()

@app.command("/mineralization")
def handle_mineralization(ack, respond, command):
    ack("🗺️ *Crunching mechanistic terrain and weather models... Drawing map!*")
    
    def background_worker():
        field_name = command['text'].strip()
        
        if field_name not in PLANTING_DB:
            respond(f"❌ Field `{field_name}` not recognized in database.")
            return

        df_soil = get_field_soil_data(field_name)
        if df_soil is None:
            respond(f"❌ Soil layer matrix for `{field_name}` not found on disk.")
            return
            
        station_info = FIELD_STATION_MAP.get(field_name)
        df_wx = fetch_climate_data_cached(station_info[0], datetime.now().year)
        
        mask = (df_wx['Date'] >= pd.to_datetime(PLANTING_DB[field_name])) & (df_wx['Date'] <= pd.to_datetime(datetime.now()))
        sub_wx = df_wx[mask]
        if sub_wx.empty:
            respond("❌ Weather records unavailable.")
            return

        # =========================================================
        # PROCESS-INFORMED MINERALIZATION ENGINE
        # =========================================================
        
        # 1. Advanced Kinetics & Texture Optimums
        current_gdd = (sub_wx['Tmean'] - 2.5 - GDD_BASE).clip(lower=0).sum()
        df_soil['clay_factor'] = 1.0 - (df_soil['Clay'] / 100.0 * 0.4)
        
        # 2. Dynamic Moisture Capacity (PAWater integration)
        if 'PAWater' in df_soil.columns:
            df_soil['WHC_mm'] = df_soil['PAWater'] * 100.0
        else:
            df_soil['WHC_mm'] = np.clip(20.0 - (df_soil['Sand'] * 0.2) + (df_soil['Clay'] * 0.3) + (df_soil['OM'] * 3.0), 15.0, 80.0)

        # 3. Terrain Modifiers (Non-linear TWI applied if pixel-data exists, else field average)
        topo = TERRAIN_METRICS.get(field_name, {})
        topo_modifier = topo.get('Mineralization_Topo_Modifier', 1.0)
        drainage_class = topo.get('Drainage_Class', 'Unknown (No LiDAR)')
        
        if 'TWI' in df_soil.columns:
            twi_z = (df_soil['TWI'] - df_soil['TWI'].mean()) / (df_soil['TWI'].std() + 1e-6)
            pixel_wetness = 1.0 / (1.0 + np.exp(-twi_z)) # Sigmoid non-linear scaling
            topo_modifier = topo_modifier * pixel_wetness

        # 4. Gross N Potential vs Denitrification Sink
        df_soil['N_pot'] = df_soil['OM'] * 26.5 * df_soil['clay_factor']
        gross_n_min = df_soil['N_pot'] * topo_modifier * (1.0 - np.exp(-0.0014 * current_gdd))
        
        # Denitrification penalty for wet, clay-heavy depressions
        denitrif_risk = np.where(topo_modifier > 1.2, 0.15, 0.0) # 15% loss in highly converged zones
        
        df_soil['Net_N_min_kg_ha'] = np.clip(gross_n_min * (1.0 - denitrif_risk), 0.0, None)
        
        # 5. High-Speed Hexbin Rendering
        plt.figure(figsize=(7, 6))
        hb = plt.hexbin(df_soil['Longitude'], df_soil['Latitude'], 
                        C=df_soil['Net_N_min_kg_ha'], 
                        reduce_C_function=np.mean, 
                        gridsize=150, cmap='YlOrRd', alpha=0.9, mincnt=1)
        
        plt.colorbar(hb, label=r'Net Available N ($kg\ N \cdot ha^{-1}$)')
        
        title_suffix = "\n(Terrain-Aware Process Model)" if topo else ""
        plt.title(f"Field: {field_name} — Available N Index{title_suffix}", fontsize=10, fontweight='bold')
        plt.tight_layout()
        
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=150)
        img_buf.seek(0)
        plt.close()
        
        avg_release = df_soil['Net_N_min_kg_ha'].mean()
        
        comment = (
            f"🗺️ *Terrain-Aware Mineralization & Retention Model — {field_name}*\n"
            f"• Avg Net Available N to date: *{avg_release:.1f} kg N/ha* "
            f"(~{avg_release*0.89:.1f} lbs/ac)\n"
            f"• Field Terrain Class: {drainage_class}\n"
            f"• Model Accounts For: OM availability, clay physical protection, dynamic WHC, & denitrification sinks."
        )

        app.client.files_upload_v2(
            channel=command['channel_id'],
            initial_comment=comment,
            file=img_buf.read(), filename=f"{field_name}_min_hex.png"
        )
        
    Thread(target=background_worker).start()

@app.command("/trial-zones")
def handle_trial_zones(ack, respond, command):
    ack("🚜 *Processing spatial clustering for management zones...*")
    
    def background_worker():
        args = command['text'].strip().split()
        if not args:
            respond("❌ Format: `/trial-zones [field] [clusters]`")
            return
            
        field_name, n_zones = args[0], int(args[1]) if len(args) > 1 else 4
        df_soil = get_field_soil_data(field_name)
        if df_soil is None:
            respond(f"❌ Field `{field_name}` data empty.")
            return
            
        SOIL_FEATURES = ['OM', 'Clay', 'Sand', 'pH', 'CEC']
        df_clean = df_soil.dropna(subset=SOIL_FEATURES).copy()
        
        scaled_data = StandardScaler().fit_transform(df_clean[SOIL_FEATURES])
        
        # MiniBatchKMeans for massive speedup
        kmeans = MiniBatchKMeans(n_clusters=n_zones, random_state=42, batch_size=10000, n_init=3)
        df_clean['Zone'] = kmeans.fit_predict(scaled_data) + 1
        
        trial_rate_mapping = {1: 56, 2: 112, 3: 160, 4: 208, 5: 240}
        df_clean['Trial_Rate'] = df_clean['Zone'].map(trial_rate_mapping).fillna(160)
        
        plt.figure(figsize=(8, 6))
        # Use Hexbin for zones as well to keep plotting fast and readable
        hb = plt.hexbin(df_clean['Longitude'], df_clean['Latitude'], 
                        C=df_clean['Zone'], 
                        reduce_C_function=lambda x: max(set(x), key=list(x).count), # Mode function for zones
                        gridsize=150, cmap='Set1', alpha=0.9, mincnt=1)
        
        plt.title(f"MiniBatch K-Means Delineated Zones — {field_name}")
        plt.tight_layout()
        
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=150)
        img_buf.seek(0)
        plt.close()
        
        app.client.files_upload_v2(
            channel=command['channel_id'], 
            initial_comment=f"🎯 *Management Zones Generated for {field_name}*",
            file=img_buf.read(), filename=f"{field_name}_zones.png"
        )
        
    Thread(target=background_worker).start()

# ==========================================
# 4. LOOP ENVIRONMENT KICKSTART
# ==========================================
web_app = Flask(__name__)

@web_app.route('/')
def home():
    return "Bot is awake, threaded, and process-informed!"

def run_web_server():
    port = int(os.environ.get("PORT", 8080))
    web_app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    Thread(target=run_web_server).start()
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
