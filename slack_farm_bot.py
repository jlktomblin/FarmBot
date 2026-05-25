import os
import io
import glob
import zipfile
import warnings
import traceback
from datetime import datetime
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

# Thread-safe Matplotlib imports
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

import requests

from flask import Flask
from waitress import serve # Production WSGI

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURATION
# =========================================================

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")

CSV_FOLDER = "./soil_data"
GDD_BASE = 5.0
MAX_CACHE_SIZE = 20 # Prevents memory leaks in production

# =========================================================
# SCIENTIFIC MODEL CONSTANTS
# =========================================================

LABILE_FRACTION = 0.20
RECALCITRANT_FRACTION = 0.80

K_FAST_BASE = 0.012
K_SLOW_BASE = 0.0014

Q10 = 2.0
T_REF = 20.0

CLAY_PROTECTION_COEFF = 0.45

DENITRIFICATION_WETNESS_THRESHOLD = 1.15
MAX_DENITRIFICATION_LOSS = 0.18

IMMOBILIZATION_CARBON_RATIO = 35

# =========================================================
# DATABASES
# =========================================================

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

# =========================================================
# INITIALIZATION & CACHE MANAGEMENT
# =========================================================

if not os.path.exists(CSV_FOLDER):
    os.makedirs(CSV_FOLDER)

for zf in glob.glob("*.zip"):
    with zipfile.ZipFile(zf, 'r') as zip_ref:
        zip_ref.extractall(CSV_FOLDER)
    os.remove(zf)

print("⚡ Building parquet cache...")

FILE_MAP = {}
SOIL_CACHE = {}
WEATHER_CACHE = {}

def enforce_cache_limit(cache_dict):
    """Evicts oldest entries to prevent memory leaks."""
    if len(cache_dict) > MAX_CACHE_SIZE:
        oldest_key = next(iter(cache_dict))
        del cache_dict[oldest_key]

def convert_to_parquet(csv_path):
    base = os.path.basename(csv_path).replace('.csv', '')
    mapped = FIELD_NAME_MAP.get(base, base)
    parquet_path = csv_path.replace('.csv', '.parquet')
    
    if not os.path.exists(parquet_path):
        try:
            df = pd.read_csv(csv_path)
            df.to_parquet(parquet_path)
        except Exception as e:
            print(f"⚠️ Failed to convert {csv_path}: {e}")
            
    return mapped, parquet_path

csv_files = glob.glob(os.path.join(CSV_FOLDER, '*.csv'))
if csv_files:
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(convert_to_parquet, csv_files))
    for mapped, pq in results:
        FILE_MAP.setdefault(mapped, []).append(pq)

print(f"✅ {len(FILE_MAP)} field datasets indexed")

TERRAIN_METRICS = {}
terrain_csv = "field_terrain_metrics.csv"
if os.path.exists(terrain_csv):
    tm = pd.read_csv(terrain_csv)
    TERRAIN_METRICS = tm.set_index('Field').to_dict('index')
    print(f"✅ Terrain metrics loaded")

# =========================================================
# DATA ACCESS
# =========================================================

def get_field_soil_data(field_name):
    if field_name in SOIL_CACHE:
        return SOIL_CACHE[field_name]

    file_paths = FILE_MAP.get(field_name)
    if not file_paths:
        return None

    dfs = []
    for fp in file_paths:
        df = pd.read_parquet(fp)
        df['Field'] = field_name
        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)
    SOIL_CACHE[field_name] = result
    enforce_cache_limit(SOIL_CACHE)
    
    return result

def fetch_climate_data(station_id, year):
    url = (f'https://climate.weather.gc.ca/climate_data/bulk_data_e.html'
           f'?format=csv&stationID={station_id}&Year={year}&Month=1&Day=1&timeframe=2&submit=Download+Data')
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        df.columns = df.columns.str.strip()

        date_col = [c for c in df.columns if 'Date' in c][0]
        tmean_col = [c for c in df.columns if 'Mean Temp' in c or 'TMEAN' in c][0]
        tmax_col = [c for c in df.columns if 'Max Temp' in c or 'TMAX' in c][0]
        tmin_col = [c for c in df.columns if 'Min Temp' in c or 'TMIN' in c][0]
        precip_col = [c for c in df.columns if 'Precip' in c or 'Rain' in c][0]

        df = df.rename(columns={
            date_col: 'Date', tmean_col: 'Tmean', tmax_col: 'Tmax', 
            tmin_col: 'Tmin', precip_col: 'Precip'
        })
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        for col in ['Tmean', 'Tmax', 'Tmin', 'Precip']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        return df.dropna(subset=['Date']).sort_values('Date')
    except Exception:
        return None

def fetch_climate_data_cached(station_id, year):
    cache_key = f"{station_id}_{year}"
    now = datetime.now()

    if cache_key in WEATHER_CACHE:
        age_hours = (now - WEATHER_CACHE[cache_key]['fetched_at']).total_seconds() / 3600
        if age_hours < 6:
            return WEATHER_CACHE[cache_key]['data']

    data = fetch_climate_data(station_id, year)
    if data is not None:
        WEATHER_CACHE[cache_key] = {'data': data, 'fetched_at': now}
        enforce_cache_limit(WEATHER_CACHE)
        
    return data

# =========================================================
# SLACK APP
# =========================================================

app = App(token=SLACK_BOT_TOKEN)

# =========================================================
# MINERALIZATION COMMAND
# =========================================================

@app.command("/mineralization")
def handle_mineralization(ack, respond, command):
    ack("🧠 Running advanced biophysical mineralization model...")

    def background_worker():
        try:
            field_name = command['text'].strip()
            planting_date_str = PLANTING_DB.get(field_name)

            if not planting_date_str:
                respond(f"⚠️ `{field_name}` has no planting date set")
                return

            station_info = FIELD_STATION_MAP.get(field_name)
            if not station_info:
                respond(f"❌ No weather station configured for `{field_name}`")
                return

            df_soil = get_field_soil_data(field_name)
            if df_soil is None or df_soil.empty:
                respond(f"❌ Soil data missing for `{field_name}`")
                return

            station_id, station_name = station_info
            df_wx = fetch_climate_data_cached(station_id, datetime.now().year)

            if df_wx is None or df_wx.empty:
                respond("❌ Weather data unavailable")
                return

            planting_date = pd.to_datetime(planting_date_str)
            sub_wx = df_wx[(df_wx['Date'] >= planting_date) & (df_wx['Date'] <= datetime.now())]

            if sub_wx.empty:
                respond("❌ No weather records available since planting")
                return

            # =====================================================
            # SOIL ARRAYS
            # =====================================================
            om = df_soil['OM'].to_numpy(dtype=np.float32)
            clay = df_soil['Clay'].to_numpy(dtype=np.float32)
            sand = df_soil['Sand'].to_numpy(dtype=np.float32)
            lon = df_soil['Longitude'].to_numpy(dtype=np.float32)
            lat = df_soil['Latitude'].to_numpy(dtype=np.float32)

            if 'PAWater' in df_soil.columns:
                whc_mm = df_soil['PAWater'].fillna(0).to_numpy(dtype=np.float32) * 100.0
            else:
                whc_mm = np.clip(20.0 - (sand * 0.2) + (clay * 0.3) + (om * 3.0), 15.0, 80.0)

            field_capacity_proxy = whc_mm * 2.5

            # =====================================================
            # TERRAIN MODIFIERS
            # =====================================================
            topo = TERRAIN_METRICS.get(field_name, {})
            base_topo_modifier = topo.get('Mineralization_Topo_Modifier', 1.0)
            drainage_class = topo.get('Drainage_Class', 'Unknown')

            if 'TWI' in df_soil.columns:
                twi = df_soil['TWI'].to_numpy(dtype=np.float32)
                twi_z = (twi - np.mean(twi)) / (np.std(twi) + 1e-6)
                wetness_index = 1.0 / (1.0 + np.exp(-twi_z))
                topo_modifier = base_topo_modifier * wetness_index
            else:
                topo_modifier = np.full(len(df_soil), base_topo_modifier, dtype=np.float32)

            # =====================================================
            # DAILY TIMESTEP MINERALIZATION KINETICS
            # =====================================================
            eff_k_fast = np.zeros(len(df_soil), dtype=np.float32)
            eff_k_slow = np.zeros(len(df_soil), dtype=np.float32)
            soil_moisture_mm = field_capacity_proxy.copy()
            total_gdd = 0

            for _, row in sub_wx.iterrows():
                tmean = row['Tmean'] - 2.5
                precip = row['Precip']
                
                # Simple ET proxy based on temperature
                et = max(0.5, tmean * 0.18) 
                
                # Daily moisture balance
                soil_moisture_mm = np.clip(soil_moisture_mm + precip - et, 1.0, field_capacity_proxy)
                moisture_ratio = soil_moisture_mm / field_capacity_proxy
                
                moisture_factor = np.where(
                    moisture_ratio < 0.5, moisture_ratio / 0.5,
                    np.where(moisture_ratio > 1.2, np.maximum(0.6, 1.2 / moisture_ratio), 1.0)
                )
                
                q10_factor = Q10 ** ((tmean - T_REF) / 10.0)
                gdd_day = max(0, tmean - GDD_BASE)
                total_gdd += gdd_day
                
                # Accumulate daily effective k
                eff_k_fast += K_FAST_BASE * q10_factor * gdd_day * moisture_factor
                eff_k_slow += K_SLOW_BASE * q10_factor * gdd_day * moisture_factor

            # Calculate Pool Fractions
            n_potential_total = om * 26.5
            n_fast_pool = n_potential_total * LABILE_FRACTION
            n_slow_pool = n_potential_total * RECALCITRANT_FRACTION
            clay_protection = np.clip(1.0 - (clay / 100.0 * CLAY_PROTECTION_COEFF), 0.45, 1.0)

            # Final Mineralization amounts
            n_min_fast = n_fast_pool * (1.0 - np.exp(-eff_k_fast * topo_modifier))
            n_min_slow = n_slow_pool * clay_protection * (1.0 - np.exp(-eff_k_slow * topo_modifier))
            gross_n_min = n_min_fast + n_min_slow

            # =====================================================
            # LOSSES & IMMOBILIZATION
            # =====================================================
            denit_loss = np.where(
                topo_modifier > DENITRIFICATION_WETNESS_THRESHOLD,
                np.minimum(MAX_DENITRIFICATION_LOSS, (topo_modifier - 1.0) * 0.12),
                0.0
            )

            net_n_min = gross_n_min * (1.0 - denit_loss)

            if 'Residue_C_N' in df_soil.columns:
                residue_cn = df_soil['Residue_C_N'].fillna(20).to_numpy(dtype=np.float32)
                immobilization_factor = np.where(residue_cn > IMMOBILIZATION_CARBON_RATIO, 0.82, 1.0)
                net_n_min *= immobilization_factor

            net_n_min = np.clip(net_n_min, 0.0, None)
            
            mean_n = float(np.mean(net_n_min))
            ci95 = 1.96 * (float(np.std(net_n_min)) / np.sqrt(len(net_n_min)))

            # =====================================================
            # THREAD-SAFE VISUALIZATION (OO API)
            # =====================================================
            fig = Figure(figsize=(10, 8))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            hb = ax.hexbin(
                lon, lat, C=net_n_min, reduce_C_function=np.mean,
                gridsize=180, cmap='YlOrRd', mincnt=1, linewidths=0.0, rasterized=True
            )
            
            cb = fig.colorbar(hb, ax=ax)
            cb.set_label('Available Mineralized N (kg N/ha)', fontsize=11)
            
            ax.set_title(f"{field_name} — Terrain-Aware Nitrogen Mineralization", fontsize=15, fontweight='bold')
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(alpha=0.15)

            fig.tight_layout()
            img_buf = io.BytesIO()
            canvas.print_png(img_buf)
            img_buf.seek(0)

            comment = (
                f"🧠 *Advanced Mineralization Report — {field_name}*\n"
                f"• Estimated Available N: *{mean_n:.1f} ± {ci95:.1f} kg N/ha*\n"
                f"• Heat Accumulation: *{total_gdd:.0f} GDD*\n"
                f"• Terrain Class: *{drainage_class}*\n"
                f"• Model Includes: Daily timestep kinetics, dynamic moisture balance, clay protection, Q10 scaling, and topographic wetness."
            )

            app.client.files_upload_v2(
                channel=command['channel_id'],
                initial_comment=comment,
                file=img_buf.read(),
                filename=f"{field_name}_mineralization.png"
            )

        except Exception as e:
            error_trace = traceback.format_exc()
            respond(f"❌ *Fatal Error during Mineralization Calculation:*\n```{str(e)}```\nCheck the server logs for the full traceback.")
            print(error_trace)

    Thread(target=background_worker).start()

# =========================================================
# WEB SERVER & STARTUP (RENDER SURVIVAL HACK)
# =========================================================

# This dummy server only exists to satisfy Render's port requirement
dummy_app = Flask(__name__)

@dummy_app.route('/')
def home():
    return "Bot is running silently in the background!"

if __name__ == "__main__":
    # 1. Start the Slack Bot in a background thread
    if SLACK_APP_TOKEN and SLACK_BOT_TOKEN:
        print("✅ Slack tokens found. Starting bot...")
        handler = SocketModeHandler(app, SLACK_APP_TOKEN)
        Thread(target=handler.start, daemon=True).start()
    else:
        print("⚠️ Missing Slack tokens. Bot cannot start.")

    # 2. Run the dummy server on the MAIN thread to prevent Render from killing the app
    port = int(os.environ.get("PORT", 8080))
    print(f"✅ Opening port {port} for Render...")
    serve(dummy_app, host="0.0.0.0", port=port)

