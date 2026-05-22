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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from flask import Flask
from threading import Thread

# ==========================================
# 1. CONFIGURATION & CREDENTIAL SECURITY
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
# 2. RUNTIME EXTRACTION & LAZY LOADING CACHE
# ==========================================
if not os.path.exists("soil_data"):
    os.makedirs("soil_data")

for zf in glob.glob("*.zip"):
    with zipfile.ZipFile(zf, 'r') as zip_ref:
        zip_ref.extractall("soil_data")
    os.remove(zf)

# Build a lightweight map of file paths instead of loading all data into RAM
FILE_MAP = {}
for f in glob.glob(os.path.join(CSV_FOLDER, '*.csv')):
    base = os.path.basename(f).replace('.csv', '')
    mapped = FIELD_NAME_MAP.get(base, base)
    # Handle cases where multiple CSVs map to one field
    if mapped not in FILE_MAP:
        FILE_MAP[mapped] = [f]
    else:
        FILE_MAP[mapped].append(f)

# Load Terrain Metrics
TERRAIN_METRICS = {}
terrain_csv = "field_terrain_metrics.csv"
if os.path.exists(terrain_csv):
    tm = pd.read_csv(terrain_csv).set_index('Field')
    TERRAIN_METRICS = tm.to_dict('index')
    print(f"✅ Terrain metrics loaded for {len(TERRAIN_METRICS)} fields")
else:
    print("⚠️  No terrain metrics found. Proceeding without topography modifiers.")

WEATHER_CACHE = {}

def get_field_soil_data(field_name):
    """Lazy-loads the soil data from disk only when requested to save RAM."""
    file_paths = FILE_MAP.get(field_name)
    if not file_paths:
        return None
    
    dfs = []
    for fp in file_paths:
        df_temp = pd.read_csv(fp)
        df_temp['Field'] = field_name
        dfs.append(df_temp)
    
    return pd.concat(dfs, ignore_index=True) if dfs else None

app = App(token=SLACK_BOT_TOKEN)
# ==========================================
# 3. INTERACTIVE SLACK ROUTINES
# ==========================================
@app.command("/gdd-chu")
def handle_gdd_chu(ack, respond, command):
    ack()
    field_name = command['text'].strip()
    if field_name not in PLANTING_DB:
        respond(f"❌ Field `{field_name}` not recognized in database.")
        return
    if PLANTING_DB[field_name] is None:
        respond(f"⚠️ *Field Notification:* `{field_name}` does not have an actual planting date recorded yet.")
        return
        
    station_info = FIELD_STATION_MAP.get(field_name)
    station_id, station_name = station_info
    p_date = pd.to_datetime(PLANTING_DB[field_name])
    current_date = datetime.now()
    
    df_curr = fetch_climate_data_cached(station_id, current_date.year)
    df_prev = fetch_climate_data_cached(station_id, current_date.year - 1)
    
    if df_curr is None or df_prev is None:
        respond("❌ Weather data down or unreachable.")
        return

    def compute_metrics(df_year, start_dt, end_dt):
        mask = (df_year['Date'] >= start_dt) & (df_year['Date'] <= end_dt)
        sub = df_year[mask].copy()
        if sub.empty: return 0, 0
        tmax_c = [c for c in sub.columns if 'Max Temp' in c or 'TMAX' in c][0]
        tmin_c = [c for c in sub.columns if 'Min Temp' in c or 'TMIN' in c][0]
        tmean_c = [c for c in sub.columns if 'Mean Temp' in c or 'TMEAN' in c][0]
        sub['Tmean'] = pd.to_numeric(sub[tmean_c], errors='coerce')
        sub['Tmax'] = pd.to_numeric(sub[tmax_c], errors='coerce')
        sub['Tmin'] = pd.to_numeric(sub[tmin_c], errors='coerce')
        gdd = (sub['Tmean'] - GDD_BASE).clip(lower=0).sum()
        def calc_chu_element(tmax, tmin):
            tmax = min(tmax, 30.0)
            ymax = 3.33 * (tmax - 10.0) - 0.084 * (tmax - 10.0)**2 if tmax > 10.0 else 0.0
            ymin = 1.8 * (tmin - 4.44) if tmin > 4.44 else 0.0
            return max(0.0, (ymax + ymin) / 2.0)
        chu = sub.apply(lambda r: calc_chu_element(r['Tmax'], r['Tmin']) if pd.notna(r['Tmax']) else 0, axis=1).sum()
        return round(gdd, 0), round(chu, 0)

    gdd_c, chu_c = compute_metrics(df_curr, p_date, pd.to_datetime(current_date.strftime('%Y-%m-%d')))
    gdd_p, chu_p = compute_metrics(df_prev, p_date - pd.DateOffset(years=1), pd.to_datetime(current_date.strftime('%Y-%m-%d')) - pd.DateOffset(years=1))
    
    plt.figure(figsize=(7, 4))
    df_curr_filtered = df_curr[(df_curr['Date'] >= p_date) & (df_curr['Date'] <= pd.to_datetime(current_date.strftime('%Y-%m-%d')))].copy()
    if not df_curr_filtered.empty:
        tmean_col = [c for c in df_curr_filtered.columns if 'Mean Temp' in c][0]
        df_curr_filtered['Tmean'] = pd.to_numeric(df_curr_filtered[tmean_col], errors='coerce')
        df_curr_filtered['GDD_cum'] = (df_curr_filtered['Tmean'] - GDD_BASE).clip(lower=0).cumsum()
        plt.plot(df_curr_filtered['Date'], df_curr_filtered['GDD_cum'], color='#4a9e6b', linewidth=2.5)
        
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

@app.command("/mineralization")
def handle_mineralization(ack, respond, command):
    ack()
    field_name = command['text'].strip()
    
    if field_name not in PLANTING_DB:
        respond(f"❌ Field `{field_name}` not recognized in database.")
        return
    if PLANTING_DB[field_name] is None:
        respond(f"⚠️ *Field Notification:* `{field_name}` does not have an actual planting date recorded yet.")
        return

    df_soil = get_field_soil_data(field_name).copy()
    if df_soil is None:
        respond(f"❌ Soil layer matrix for `{field_name}` not in memory cache.")
        return
        
    station_info = FIELD_STATION_MAP.get(field_name)
    df_wx = fetch_climate_data_cached(station_info[0], datetime.now().year)
    
    mask = (df_wx['Date'] >= pd.to_datetime(PLANTING_DB[field_name])) & (df_wx['Date'] <= pd.to_datetime(datetime.now()))
    sub_wx = df_wx[mask].copy()
    
    if sub_wx.empty:
        respond("❌ Weather records for tracking window are currently unavailable.")
        return

    tmean_col = [c for c in sub_wx.columns if 'Mean Temp' in c or 'TMEAN' in c][0]
    precip_col = [c for c in sub_wx.columns if 'Total Precip' in c or 'Total Rain' in c or 'PRECIP' in c][0]
    
    sub_wx['Tmean'] = pd.to_numeric(sub_wx[tmean_col], errors='coerce').fillna(12.0)
    sub_wx['Precip'] = pd.to_numeric(sub_wx[precip_col], errors='coerce').fillna(0.0)

    # Calculate GDD for basic kinetic progression
    current_gdd = (sub_wx['Tmean'] - 2.5 - GDD_BASE).clip(lower=0).sum()

    # =========================================================
    # PROCESS-INFORMED MINERALIZATION ENGINE (Terrain & Texture)
    # =========================================================
    
    # 1. Topographic Context
    topo = TERRAIN_METRICS.get(field_name, {})
    topo_modifier  = topo.get('Mineralization_Topo_Modifier', 1.0)
    drainage_class = topo.get('Drainage_Class', 'Unknown (No LiDAR)')
    elev_range     = topo.get('Elevation_Range_m', 'N/A')

    # 2. Advanced Kinetic Math
    df_soil['clay_factor'] = 1.0 - (df_soil['Clay'] / 100.0 * 0.4)
    df_soil['N_pot'] = df_soil['OM'] * 26.5 * df_soil['clay_factor']
    
    df_soil['Net_N_min_kg_ha'] = (df_soil['N_pot'] 
                                  * topo_modifier 
                                  * (1.0 - np.exp(-0.0014 * current_gdd)))
    
    df_soil['Net_N_min_kg_ha'] = np.clip(df_soil['Net_N_min_kg_ha'], 0.0, None)
    
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(df_soil['Longitude'], df_soil['Latitude'], c=df_soil['Net_N_min_kg_ha'], cmap='YlOrRd', s=8, alpha=0.8)
    plt.colorbar(sc, label='Net Mineralized N ($kg\ N \cdot ha^{-1}$)')
    
    title_suffix = "\n(Topography/LiDAR Integrated)" if topo else ""
    plt.title(f"Field: {field_name} — Net N Mineralization Index{title_suffix}", fontsize=10, fontweight='bold')
    plt.tight_layout()
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=150)
    img_buf.seek(0)
    plt.close()
    
    avg_release = df_soil['Net_N_min_kg_ha'].mean()
    
    comment = (
        f"🗺️ *Mineralization Map — {field_name}*\n"
        f"• Avg N released to date: *{avg_release:.1f} kg N/ha* "
        f"(~{avg_release*0.89:.1f} lbs/ac)\n"
        f"• Heat accumulation: {current_gdd:.0f} GDD since planting\n"
        f"• Terrain: {drainage_class} | Relief: {elev_range}m\n"
        f"• Topo drainage modifier applied: {topo_modifier:.2f}x\n"
        f"• Sub-field variation driven by SoilOptix OM × clay protection × topography"
    )

    app.client.files_upload_v2(
        channel=command['channel_id'],
        initial_comment=comment,
        file=img_buf.read(), filename=f"{field_name}_min.png"
    )

@app.command("/trial-zones")
def handle_trial_zones(ack, respond, command):
    ack()
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
    df_clean['Zone'] = KMeans(n_clusters=n_zones, random_state=42, n_init=10).fit_predict(scaled_data) + 1
    
    trial_rate_mapping = {1: 56, 2: 112, 3: 160, 4: 208, 5: 240}
    df_clean['Trial_Prescription_Rate_kg_ha'] = df_clean['Zone'].map(trial_rate_mapping).fillna(160)
    
    plt.figure(figsize=(8, 6))
    colors_palette = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3']
    for z in sorted(df_clean['Zone'].unique()):
        sub_z = df_clean[df_clean['Zone'] == z]
        plt.scatter(sub_z['Longitude'], sub_z['Latitude'], label=f"Zone {z} — {trial_rate_mapping.get(z, 160)} kg/ha", color=colors_palette[(z-1) % 5], s=6, alpha=0.7)
    plt.title(f"K-Means Delineated On-Farm N Trial Zones — Field: {field_name}")
    plt.legend(title='Management Strips', loc='best', fontsize=8)
    plt.tight_layout()
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=150)
    img_buf.seek(0)
    plt.close()
    
    csv_buf = io.StringIO()
    df_clean[['Longitude', 'Latitude', 'Zone', 'Trial_Prescription_Rate_kg_ha']].to_csv(csv_buf, index=False)
    
    app.client.files_upload_v2(channel=command['channel_id'], file=img_buf.read(), filename=f"{field_name}_trial_zones.png")
    app.client.files_upload_v2(channel=command['channel_id'], content=csv_buf.getvalue(), filename=f"{field_name}_prescription_points.csv")

# ==========================================
# 4. LOOP ENVIRONMENT KICKSTART
# ==========================================
web_app = Flask(__name__)

@web_app.route('/')
def home():
    return "Bot is awake and listening!"

def run_web_server():
    port = int(os.environ.get("PORT", 8080))
    web_app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    Thread(target=run_web_server).start()
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
