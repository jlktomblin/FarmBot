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

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURATION
# =========================================================

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")

CSV_FOLDER = "./soil_data"
GDD_BASE = 10.0
MAX_CACHE_SIZE = 38 # Caches all fields (Monitor Render RAM usage!)

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
    'Campbell': '2026-05-15', 'Bercab 1': '2026-05-15', 'Bercab 2': '2026-05-17', 
    'Sydenham 2 North': '2026-05-18', 'Sydenham 2 South': '2026-05-18', 'Clare Horst': '2026-05-21',
    'Marvara / Judd': '2026-05-22', 'Biermans': '2026-05-22', 'FieldAndFlock 2': '2026-05-19',
    
    # Unplanted or fields without data
    'Sydenham 1': None, 'McAlpine': None
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

    # =====================================================
    # SARNIA / LAMBTON REGION
    # =====================================================

    'Bercab 1': (48549, 'WALLACEBURG CDA'),
    'Bercab 2': (48549, 'WALLACEBURG CDA'),
    'Burm': (48549, 'WALLACEBURG CDA'),
    'Sydenham 1': (48549, 'WALLACEBURG CDA'),
    'Sydenham 2 North': (48549, 'WALLACEBURG CDA'),
    'Sydenham 2 South': (48549, 'WALLACEBURG CDA'),
    'Gerrits': (48549, 'WALLACEBURG CDA'),
    'Kerrington': (48549, 'WALLACEBURG CDA'),
    'McAlpine': (48549, 'WALLACEBURG CDA'),

    # =====================================================
    # NORFOLK / DELHI REGION
    # =====================================================

    'Campbell': (27528, 'DELHI CS'),
    'FieldAndFlock 1': (27528, 'DELHI CS'),
    'FieldAndFlock 2': (27528, 'DELHI CS'),
    'Moosberger 1': (27528, 'DELHI CS'),
    'Moosberger 2': (27528, 'DELHI CS'),
    'JD Peters': (27528, 'DELHI CS'),

    # =====================================================
    # BRANT / BRANTFORD REGION
    # =====================================================

    'Harrison Farms': (48532, 'WOODSTOCK'),
    'Veldale': (48532, 'WOODSTOCK'),
    'Triple Lane Farms': (48532, 'WOODSTOCK'),


    # =====================================================
    # STRATFORD / WATERLOO-WELLINGTON
    # =====================================================

    'Schumhaven': (4823, 'STRATFORD AUTO'),
    'Leis': (4823, 'STRATFORD AUTO'),
    'Gerber Acres': (4823, 'STRATFORD AUTO'),
    'Benderbrook 1': (4823, 'STRATFORD AUTO'),
    'Benderbrook 2': (4823, 'STRATFORD AUTO'),

    # =====================================================
    # ELORA CLUSTER
    # =====================================================

    'Clare Horst': (41983, 'ELORA RCS'),
    'Marvara / Judd': (41983, 'ELORA RCS'),
    'Klavan': (41983, 'ELORA RCS'),
    'Triaro': (41983, 'ELORA RCS'),
}
    # =====================================================
    # LAKE HURON SHORELINE
    # =====================================================

    'Wettlaufer': (27529, 'GODERICH CLIMATE'),
    'Brucelea': (27529, 'GODERICH CLIMATE'),

    # =====================================================
    # MIDWEST / WINGHAM / BRUCE
    # =====================================================

    'Renwick 2': (48569, 'WINGHAM AUTO'),

    'Lang': (48568, 'CHESLEY CLIMATE'),
    'Biermans': (48568, 'CHESLEY CLIMATE'),
    'Christie-1': (48568, 'CHESLEY CLIMATE'),
    'Christie-2': (48568, 'CHESLEY CLIMATE'),
    'Highland': (48568, 'CHESLEY CLIMATE'),

    'Renwick 1': (7844, 'MOUNT FOREST AUT'),
    'Schaus': (7844, 'MOUNT FOREST AUT'),
    'GerMar Farms (Grubb)': (7844, 'MOUNT FOREST AUT'),

    # =====================================================
    # WINDSOR / ESSEX
    # =====================================================

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
    print("⏳ Processing files sequentially to save memory...")
    for f in csv_files:
        mapped, pq = convert_to_parquet(f)
        FILE_MAP.setdefault(mapped, []).append(pq)

print(f"✅ {len(FILE_MAP)} field datasets indexed")

TERRAIN_METRICS = {}
terrain_csv = "field_terrain_metrics.csv"
if os.path.exists(terrain_csv):
    tm = pd.read_csv(terrain_csv)
    TERRAIN_METRICS = tm.set_index('Field').to_dict('index')
    print("✅ Terrain metrics loaded")

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
# COMMAND HANDLERS
# =========================================================

@app.command("/mineralization")
def handle_mineralization(ack, respond, command):
    ack("🧠 Running advanced APSIM-style biophysical mineralization model...")

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
            # SCIENTIFICALLY IMPROVED N MINERALIZATION MODEL (APSIM-STYLE)
            # =====================================================
            # Organic matter pools (Calibrated to true Soil Total N mass)
            ACTIVE_POOL_FRAC = 0.015  # 1.5% of total N (Microbial biomass & fresh residue)
            SLOW_POOL_FRAC   = 0.485  # 48.5% of total N (Humified particulate OM)
            PASSIVE_POOL_FRAC = 0.50  # 50.0% of total N (Chemically protected)

            # Daily decay rates (Calibrated for APSIM-style base mass)
            K_ACTIVE = 0.015   # 1.5% turnover per day at optimal temp/moisture
            K_SLOW   = 0.0004  # 0.04% turnover per day
            K_PASSIVE = 0.00001 # Micro-bleed

            Q10 = 2.0
            T_REF = 20.0
            SOIL_DEPTH_MM = 150.0  # Standard 6-inch sampling depth
            SOIL_DEPTH_CM = 15.0

            # -----------------------------------------------------
            # SOIL INPUTS & MASS BALANCE
            # -----------------------------------------------------
            om = df_soil['OM'].to_numpy(dtype=np.float32)
            clay = df_soil['Clay'].to_numpy(dtype=np.float32)
            sand = df_soil['Sand'].to_numpy(dtype=np.float32)
            lon = df_soil['Longitude'].to_numpy(dtype=np.float32)
            lat = df_soil['Latitude'].to_numpy(dtype=np.float32)

            # Pedotransfer approximation for bulk density (g/cm^3)
            bulk_density = 1.6 - (om * 0.03) - (clay * 0.002)
            bulk_density = np.clip(bulk_density, 1.0, 1.7)
            
            # Calculate total physical soil mass per hectare (kg/ha)
            soil_mass_kg_ha = bulk_density * SOIL_DEPTH_CM * 100000.0

            # Estimate soil organic carbon (%) using Van Bemmelen factor
            soc_pct = om * 0.58
            # Approximate total organic nitrogen (%)
            soil_n_pct = soc_pct / 12.0

            # Convert percentage to actual kg N/ha in the soil profile
            total_n_kg_ha = soil_mass_kg_ha * (soil_n_pct / 100.0)

            # Partition physical N pools (kg N/ha)
            n_active = total_n_kg_ha * ACTIVE_POOL_FRAC
            n_slow   = total_n_kg_ha * SLOW_POOL_FRAC
            n_passive = total_n_kg_ha * PASSIVE_POOL_FRAC

            # -----------------------------------------------------
            # WEATHER ARRAYS & WATER BALANCE
            # -----------------------------------------------------
            tmean = sub_wx['Tmean'].to_numpy(dtype=np.float32)
            precip = sub_wx['Precip'].to_numpy(dtype=np.float32)

            # Simplified PET estimate
            pet = np.maximum(1.0, tmean * 0.22)

            # Total Porosity (m3/m3) derived from bulk density
            porosity = 1.0 - (bulk_density / 2.65)
            
            # Initialize soil water at 75% of porosity (Spring baseline)
            soil_water_vwc = porosity * 0.75

            daily_n_min = np.zeros(len(df_soil), dtype=np.float32)
            total_gdd = float(np.maximum(0, tmean - GDD_BASE).sum())

            # -----------------------------------------------------
            # FAST SPATIAL SIMULATION LOOP
            # -----------------------------------------------------
            for d in range(len(tmean)):
                # Add precipitation and remove PET (converted from mm to Volumetric Fraction)
                soil_water_vwc += (precip[d] / SOIL_DEPTH_MM)
                soil_water_vwc -= (pet[d] / SOIL_DEPTH_MM)

                # Cap water between permanent wilting point and saturation (porosity)
                soil_water_vwc = np.clip(soil_water_vwc, 0.05, porosity)

                # Calculate Water-Filled Pore Space (WFPS)
                wfps = soil_water_vwc / porosity
                wfps = np.clip(wfps, 0.05, 1.0)

                # Temperature scalar
                temp_scalar = Q10 ** ((tmean[d] - T_REF) / 10.0)
                temp_scalar = np.clip(temp_scalar, 0.1, 4.0)

                # Moisture scalar (Bell curve peaking at 60% WFPS)
                moisture_scalar = np.where(
                    wfps < 0.6,
                    wfps / 0.6,
                    np.exp(-((wfps - 0.6) ** 2) / 0.08)
                )
                moisture_scalar = np.clip(moisture_scalar, 0.05, 1.0)

                # Clay protection scalar
                clay_scalar = 1.0 - (clay / 100.0 * 0.5)
                clay_scalar = np.clip(clay_scalar, 0.4, 1.0)

                # Combined Environmental Multiplier
                env = temp_scalar * moisture_scalar * clay_scalar

                # First-Order Pool Mineralization (kg N/ha)
                active_min = n_active * K_ACTIVE * env
                slow_min   = n_slow * K_SLOW * env
                passive_min = n_passive * K_PASSIVE * env

                gross_min = active_min + slow_min + passive_min

                # Denitrification triggers sharply above 75% WFPS
                denit = np.where(wfps > 0.75, (wfps - 0.75) * 0.4, 0.0)
                denit = np.clip(denit, 0.0, 0.25)

                net_min = gross_min * (1.0 - denit)

                # Deplete pools (Mass balance tracking)
                n_active -= active_min
                n_slow -= slow_min
                n_passive -= passive_min

                daily_n_min += net_min

            # -----------------------------------------------------
            # FINAL ADJUSTMENTS & IMMOBILIZATION
            # -----------------------------------------------------
            net_n_min = np.clip(daily_n_min, 0.0, None)

            if 'Residue_C_N' in df_soil.columns:
                residue_cn = df_soil['Residue_C_N'].fillna(20).to_numpy(dtype=np.float32)
                immobilization_factor = np.where(residue_cn > IMMOBILIZATION_CARBON_RATIO, 0.82, 1.0)
                net_n_min *= immobilization_factor

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
            
            ax.set_title(f"{field_name} — APSIM-Style N Mineralization", fontsize=15, fontweight='bold')
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(alpha=0.15)

            fig.tight_layout()
            img_buf = io.BytesIO()
            canvas.print_png(img_buf)
            img_buf.seek(0)

            comment = (
                f"🧠 *Advanced APSIM-Style Report — {field_name}*\n"
                f"• Estimated Available N: *{mean_n:.1f} ± {ci95:.1f} kg N/ha*\n"
                f"• Heat Accumulation: *{total_gdd:.0f} GDD*\n"
                f"• Model Includes: Active/Slow/Passive pools, WFPS moisture curve, clay protection, and 1st-order decay."
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
# GDD / CHU COMMAND (UPDATED WITH MODIFIED GDD)
# =========================================================

@app.command("/gdd-chu")
def handle_gdd_chu(ack, respond, command):
    ack("🌱 Fetching weather data and calculating Modified GDD & CHU...")

    def background_worker():
        try:
            field_name = command['text'].strip()

            if field_name not in PLANTING_DB:
                respond(f"❌ Field `{field_name}` not recognized. Check spelling.")
                return

            planting_date_str = PLANTING_DB.get(field_name)
            if not planting_date_str:
                respond(f"⚠️ `{field_name}` has no planting date set in PLANTING_DB.")
                return

            station_info = FIELD_STATION_MAP.get(field_name)
            if not station_info:
                respond(f"❌ No weather station configured for `{field_name}`.")
                return

            station_id, station_name = station_info
            planting_date = pd.to_datetime(planting_date_str)
            current_date  = datetime.now()

            # Fetch current and previous year in parallel
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_curr = ex.submit(fetch_climate_data_cached, station_id, current_date.year)
                fut_prev = ex.submit(fetch_climate_data_cached, station_id, current_date.year - 1)
                df_curr = fut_curr.result()
                df_prev = fut_prev.result()

            if df_curr is None:
                respond(f"❌ Could not fetch weather data from {station_name}.")
                return

            def compute_metrics(df_year, start_dt, end_dt):
                if df_year is None or df_year.empty:
                    return 0.0, 0.0
                mask = (df_year['Date'] >= start_dt) & (df_year['Date'] <= end_dt)
                sub  = df_year[mask]
                if sub.empty:
                    return 0.0, 0.0

                tmax = sub['Tmax'].to_numpy(dtype=np.float32)
                tmin = sub['Tmin'].to_numpy(dtype=np.float32)

                # Vectorized Modified GDD (Cap at 30, Floor at 10 BEFORE averaging)
                tmax_adj = np.clip(tmax, GDD_BASE, 30.0)
                tmin_adj = np.clip(tmin, GDD_BASE, 30.0)
                gdd = float((((tmax_adj + tmin_adj) / 2.0) - GDD_BASE).sum())

                # Vectorized CHU (OMAFRA formula)
                tmax_clipped = np.minimum(tmax, 30.0)
                ymax = np.where(tmax_clipped > 10.0,
                                3.33 * (tmax_clipped - 10.0) - 0.084 * (tmax_clipped - 10.0)**2,
                                0.0)
                ymin = np.where(tmin > 4.44, 1.8 * (tmin - 4.44), 0.0)
                chu  = float(np.maximum(0.0, (ymax + ymin) / 2.0).sum())

                return round(gdd, 0), round(chu, 0)

            # Current season from planting to today
            gdd_c, chu_c = compute_metrics(df_curr, planting_date, current_date)

            # Same period last year
            prev_start = planting_date  - pd.DateOffset(years=1)
            prev_end   = current_date   - pd.DateOffset(years=1)
            gdd_p, chu_p = compute_metrics(df_prev, prev_start, prev_end)

            # GDD trend chart
            fig = Figure(figsize=(9, 4))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            df_plot = df_curr[
                (df_curr['Date'] >= planting_date) &
                (df_curr['Date'] <= current_date)
            ].copy()

            if not df_plot.empty:
                # Vectorized cumulative plotting for Modified GDD
                tmax_adj_plot = np.clip(df_plot['Tmax'], GDD_BASE, 30.0)
                tmin_adj_plot = np.clip(df_plot['Tmin'], GDD_BASE, 30.0)
                df_plot['GDD_cum'] = (((tmax_adj_plot + tmin_adj_plot) / 2.0) - GDD_BASE).cumsum()
                
                ax.plot(df_plot['Date'], df_plot['GDD_cum'],
                        color='#4a9e6b', linewidth=2.5, label='Current Season')
                ax.fill_between(df_plot['Date'], df_plot['GDD_cum'],
                                alpha=0.15, color='#4a9e6b')

            # Last year comparison line
            if df_prev is not None:
                df_prev_plot = df_prev[
                    (df_prev['Date'] >= prev_start) &
                    (df_prev['Date'] <= prev_end)
                ].copy()
                if not df_prev_plot.empty:
                    # Vectorized cumulative plotting for LY Modified GDD
                    tmax_adj_prev = np.clip(df_prev_plot['Tmax'], GDD_BASE, 30.0)
                    tmin_adj_prev = np.clip(df_prev_plot['Tmin'], GDD_BASE, 30.0)
                    df_prev_plot['GDD_cum'] = (((tmax_adj_prev + tmin_adj_prev) / 2.0) - GDD_BASE).cumsum()
                    
                    # Shift dates forward 1 year so lines overlap on same x-axis
                    df_prev_plot['Date_shifted'] = df_prev_plot['Date'] + pd.DateOffset(years=1)
                    ax.plot(df_prev_plot['Date_shifted'], df_prev_plot['GDD_cum'],
                            color='#aaaaaa', linewidth=1.5, linestyle='--', label='Last Season')

            trend = 'ahead of' if gdd_c > gdd_p else 'behind'
            diff  = abs(gdd_c - gdd_p)
            ax.set_title(f"{field_name} — Cumulative Modified GDD Since Planting ({planting_date_str})",
                         fontsize=11, fontweight='bold')
            ax.set_ylabel(f'Accumulated GDD (Base {int(GDD_BASE)}°C)')
            ax.legend(fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.5)
            fig.autofmt_xdate()
            fig.tight_layout()

            img_buf = io.BytesIO()
            canvas.print_png(img_buf)
            img_buf.seek(0)

            # CHU maturity context
            if chu_c < 2500:
                chu_context = "Early hybrids only (<2500 CHU)"
            elif chu_c < 2800:
                chu_context = "Mid-season hybrids (2500–2800 CHU)"
            elif chu_c < 3100:
                chu_context = "Full season hybrids (2800–3100 CHU)"
            else:
                chu_context = "Long season potential (>3100 CHU)"

            comment = (
                f"🌱 *Field Weather Report — {field_name}*\n"
                f"📅 Planted: {planting_date_str} | Station: {station_name}\n"
                f"───────────────────────────────\n"
                f"🌡️ *Modified GDD (Base {int(GDD_BASE)}°C, 30°C Cap):*\n"
                f"  • This season: *{gdd_c:.0f} GDD*\n"
                f"  • Last season (same period): {gdd_p:.0f} GDD\n"
                f"  • Currently *{diff:.0f} GDD {trend} last year*\n\n"
                f"🌽 *Corn Heat Units (OMAFRA):*\n"
                f"  • This season: *{chu_c:.0f} CHU*\n"
                f"  • Last season (same period): {chu_p:.0f} CHU\n"
                f"  • Context: {chu_context}"
            )

            app.client.files_upload_v2(
                channel=command['channel_id'],
                initial_comment=comment,
                file=img_buf.read(),
                filename=f"{field_name}_gdd_chu.png"
            )

        except Exception as e:
            respond(f"❌ Error: `{str(e)}`")
            print(traceback.format_exc())

    Thread(target=background_worker).start()
    
# =========================================================
# TRIAL ZONES COMMAND
# =========================================================

@app.command("/trial-zones")
def handle_trial_zones(ack, respond, command):
    ack("🚜 Generating management zones...")

    def background_worker():
        try:
            args = command['text'].strip().split()
            if not args:
                respond("❌ Usage: `/trial-zones [FieldName] [NumberOfZones]`\n"
                        "Example: `/trial-zones Wettlaufer 4`")
                return

            field_name = args[0]
            n_zones    = int(args[1]) if len(args) > 1 else 4

            if n_zones < 2 or n_zones > 6:
                respond("⚠️ Number of zones must be between 2 and 6.")
                return

            df_soil = get_field_soil_data(field_name)
            if df_soil is None or df_soil.empty:
                respond(f"❌ No soil data found for `{field_name}`.")
                return

            SOIL_FEATURES = ['OM', 'Clay', 'Sand', 'pH', 'CEC']
            available    = [f for f in SOIL_FEATURES if f in df_soil.columns]
            if len(available) < 3:
                respond(f"❌ Not enough soil columns for clustering. Found: {available}")
                return

            df_clean = df_soil.dropna(subset=available).copy()
            if len(df_clean) < n_zones * 10:
                respond(f"⚠️ Not enough clean data points ({len(df_clean)}) for {n_zones} zones.")
                return

            scaled = StandardScaler().fit_transform(df_clean[available])

            kmeans = MiniBatchKMeans(
                n_clusters=n_zones, random_state=42,
                batch_size=10000, n_init=5
            )
            df_clean['Zone'] = kmeans.fit_predict(scaled) + 1

            # Zone summary stats
            zone_summary = df_clean.groupby('Zone')[available].mean().round(2)

            # N rate prescription — zones ranked by OM (higher OM = lower N needed)
            om_rank   = zone_summary['OM'].rank(ascending=False).astype(int)
            base_rates = {1: 180, 2: 160, 3: 140, 4: 120, 5: 100, 6: 80}
            zone_rates = {z: base_rates.get(r, 140) for z, r in om_rank.items()}
            df_clean['N_Rate_kg_ha'] = df_clean['Zone'].map(zone_rates)

            # Plot
            colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3', '#a65628']
            fig = Figure(figsize=(10, 8))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            for z in sorted(df_clean['Zone'].unique()):
                sub_z = df_clean[df_clean['Zone'] == z]
                rate  = zone_rates.get(z, 140)
                ax.scatter(sub_z['Longitude'], sub_z['Latitude'],
                           c=colors[(z-1) % len(colors)],
                           label=f"Zone {z} — {rate} kg N/ha",
                           s=4, alpha=0.7, linewidths=0)

            ax.set_title(f"{field_name} — {n_zones} Management Zones\n"
                         f"Clustered on: {', '.join(available)}",
                         fontsize=12, fontweight='bold')
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.legend(title="Zone — N Rate", fontsize=9, markerscale=3)
            ax.grid(alpha=0.2)
            fig.tight_layout()

            img_buf = io.BytesIO()
            canvas.print_png(img_buf)
            img_buf.seek(0)

            # Build zone summary text
            zone_lines = []
            for z in sorted(zone_summary.index):
                om_val  = zone_summary.loc[z, 'OM'] if 'OM' in zone_summary.columns else 'N/A'
                ph_val  = zone_summary.loc[z, 'pH'] if 'pH' in zone_summary.columns else 'N/A'
                cec_val = zone_summary.loc[z, 'CEC'] if 'CEC' in zone_summary.columns else 'N/A'
                rate    = zone_rates.get(z, 140)
                count   = len(df_clean[df_clean['Zone'] == z])
                zone_lines.append(
                    f"  Zone {z}: OM={om_val}% | pH={ph_val} | CEC={cec_val} | "
                    f"→ *{rate} kg N/ha* ({count:,} pts)"
                )

            comment = (
                f"🎯 *Management Zones — {field_name}*\n"
                f"Clustered {len(df_clean):,} soil points into {n_zones} zones "
                f"using: {', '.join(available)}\n\n"
                f"*Zone Summary:*\n" + "\n".join(zone_lines) + "\n\n"
                f"_N rates based on OM ranking — higher OM zones receive lower N input_"
            )

            app.client.files_upload_v2(
                channel=command['channel_id'],
                initial_comment=comment,
                file=img_buf.read(),
                filename=f"{field_name}_zones.png"
            )

            # Also send a CSV prescription file
            csv_buf = io.StringIO()
            df_clean[['Latitude', 'Longitude', 'Zone', 'N_Rate_kg_ha']].to_csv(
                csv_buf, index=False)
            app.client.files_upload_v2(
                channel=command['channel_id'],
                content=csv_buf.getvalue(),
                filename=f"{field_name}_prescription.csv",
                initial_comment="📄 Prescription CSV ready for your application controller."
            )

        except Exception as e:
            respond(f"❌ Error: `{str(e)}`")
            print(traceback.format_exc())

    Thread(target=background_worker).start()
# =========================================================
# HELP COMMAND
# =========================================================

@app.command("/ag-help")
def handle_help(ack, respond, command):
    ack()
    respond(
        "🌱 *Upside Agronomy Bot — Available Commands*\n"
        "─────────────────────────────────────────\n"
        "*/gdd-chu [FieldName]*\n"
        "  Cumulative GDD & CHU since planting vs last year\n"
        "  _Example: `/gdd-chu Wettlaufer`_\n\n"
        "*/mineralization [FieldName]*\n"
        "  Sub-field nitrogen mineralization map with terrain & moisture model\n"
        "  _Example: `/mineralization Wettlaufer`_\n\n"
        "*/trial-zones [FieldName] [Zones]*\n"
        "  K-Means management zones with N rate prescription CSV\n"
        "  _Example: `/trial-zones Wettlaufer 4`_\n\n"
        "*/ag-help*\n"
        "  Show this help message\n\n"
        f"📋 *Fields with planting dates configured:* "
        f"{sum(1 for v in PLANTING_DB.values() if v is not None)}/39"
    )
# =========================================================
# WEB SERVER & STARTUP
# =========================================================

dummy_app = Flask(__name__)

@dummy_app.route('/')
def home():
    return "Bot is running silently in the background!"

if __name__ == "__main__":
    if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
        print("⚠️ Missing Slack tokens — check environment variables on Render")
    else:
        print("✅ Starting Slack bot in background thread...")
        handler = SocketModeHandler(app, SLACK_APP_TOKEN)
        Thread(target=handler.start, daemon=True).start()
        print("✅ Slack bot connected")

    port = int(os.environ.get("PORT", 8080))
    print(f"✅ Starting web server on port {port}...")
    
    # Use Flask directly — no waitress dependency needed
    dummy_app.run(host="0.0.0.0", port=port, use_reloader=False)
