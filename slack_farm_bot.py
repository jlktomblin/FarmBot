import os
import io
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Thread-safe background plotting
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# ==========================================
# 1. CONFIGURATION & CORE DATA STRUCTURES
# ==========================================
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "xoxb-7042444002102-11183793169505-oaDsM00cEAJavkDEtNqsN0Bg")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN", "xapp-1-A0B59QBLHN2-11210190149504-e8eaac4413f5b0e3860426bae2a37f8fc27c6ed79b84a38fcb4f906e335b5d81")

import zipfile

# --- NEW UNZIP CODE FOR CLOUD ---
# Create the folder if it doesn't exist on the cloud computer
if not os.path.exists("soil_data"):
    os.makedirs("soil_data")

# Find every zip file we uploaded, open it, and dump the CSVs into the folder
for zf in glob.glob("*.zip"):
    print(f"📦 Unzipping {zf}...")
    with zipfile.ZipFile(zf, 'r') as zip_ref:
        zip_ref.extractall("soil_data")
    os.remove(zf)  # Clean up the zip file to save cloud space!
# --------------------------------

CSV_FOLDER = r"./soil_data"  # Path to your folder holding the 38 SoilOptix CSVs
GDD_BASE = 5.0

# 📅 Extracted from '2026 Field View 339d240931f9806ea0a4d31890304988.csv'
PLANTING_DB = {
    'Benderbrook 1': '2026-05-07',
    'Benderbrook 2': '2026-05-06',
    'Brucelea': '2026-05-06',
    'Christie-2': '2026-05-17',
    'Christie-1': '2026-05-18',
    'Leis': '2026-05-17',
    'Burm': '2026-05-13',
    'FieldAndFlock 1': '2026-05-07',
    'Gerber Acres': '2026-05-06',
    'GerMar Farms (Grubb)': '2026-05-01',
    'Gerrits': '2026-05-10',
    'Harrison Farms': '2026-05-16',
    'Highland': '2026-05-06',
    'JD Peters': '2026-05-18',
    'Kerrington': '2026-05-15',
    'Klavan': '2026-05-09',
    'Lang': '2026-05-15',
    'Moosberger 2': '2026-05-11',
    'Renwick 2': '2026-05-07',
    'Renwick 1': '2026-05-08',
    'Schaus': '2026-05-12',
    'Schumhaven': '2026-05-09',
    'Triaro': '2026-05-12',
    'Triple Lane Farms': '2026-05-11',
    'Veldale': '2026-05-12',
    'Wecker': '2026-05-08',
    'Wettlaufer': '2026-05-12',
    # Defaulting unlisted fields to May 10th (Modify as needed)
    'Bercab 1': '2026-05-10',
    'Bercab 2': '2026-05-10',
    'Sydenham 1': '2026-05-10',
    'Sydenham 2 North': '2026-05-10',
    'Sydenham 2 South': '2026-05-10',
    'McAlpine': '2026-05-10',
    'FieldAndFlock 2': '2026-05-10',
    'Moosberger 1': '2026-05-10',
    'Clare Horst': '2026-05-10',
    'Marvara / Judd': '2026-05-10',
    'Biermans': '2026-05-10',
    'Campbell': '2026-05-06'
}

# 🗺️ Full 38 Raw CSV filename-to-canonical mappings
FIELD_NAME_MAP = {
    'Clare_Horst_Home_East_NutrientTexture': 'Clare Horst',
    'Moose_CSV': 'Moosberger 1',
    'Roth_CSV': 'Benderbrook 1',
    'Tim60_CSV': 'Benderbrook 2',
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
    'Upside_Robotics_Roland_McAlpine_McAlpine_2_NutrientTexture': 'McAlpine',
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

# 📡 Full 39-Field Weather Station Dictionary
FIELD_STATION_MAP = {
    'Bercab 1': (48373, 'SARNIA'),
    'Bercab 2': (48373, 'SARNIA'),
    'Burm': (48373, 'SARNIA'),
    'Sydenham 1': (48373, 'SARNIA'),
    'Sydenham 2 North': (48373, 'SARNIA'),
    'Sydenham 2 South': (48373, 'SARNIA'),
    'Gerrits': (48373, 'SARNIA'),
    'Kerrington': (48373, 'SARNIA'),
    'McAlpine': (48373, 'SARNIA'),
    'Campbell': (27528, 'DELHI CS'),
    'FieldAndFlock 1': (27528, 'DELHI CS'),
    'FieldAndFlock 2': (27528, 'DELHI CS'),
    'Moosberger 1': (27528, 'DELHI CS'),
    'Moosberger 2': (27528, 'DELHI CS'),
    'JD Peters': (27528, 'DELHI CS'),
    'Harrison Farms': (53378, 'BRANTFORD AIRPORT'),
    'Veldale': (53378, 'BRANTFORD AIRPORT'),
    'Triple Lane Farms': (53378, 'BRANTFORD AIRPORT'),
    'Schumhaven': (10999, 'LONDON CS'),
    'Leis': (10999, 'LONDON CS'),
    'Gerber Acres': (10999, 'LONDON CS'),
    'Benderbrook 1': (10999, 'LONDON CS'),
    'Benderbrook 2': (10999, 'LONDON CS'),
    'Clare Horst': (41983, 'ELORA RCS'),
    'Marvara / Judd': (41983, 'ELORA RCS'),
    'Klavan': (41983, 'ELORA RCS'),
    'Triaro': (41983, 'ELORA RCS'),
    'Wettlaufer': (27529, 'GODERICH CLIMATE'),
    'Brucelea': (27529, 'GODERICH CLIMATE'),
    'Renwick 2': (48569, 'WINGHAM AUTO'),
    'Lang': (48568, 'CHESLEY CLIMATE'),
    'Biermans': (48568, 'CHESLEY CLIMATE'),
    'Christie-1': (48568, 'CHESLEY CLIMATE'),
    'Christie-2': (48568, 'CHESLEY CLIMATE'),
    'Highland': (48568, 'CHESLEY CLIMATE'),
    'Renwick 1': (7844, 'MOUNT FOREST AUT'),
    'Schaus': (7844, 'MOUNT FOREST AUT'),
    'GerMar Farms (Grubb)': (7844, 'MOUNT FOREST AUT'),
    'Wecker': (54738, 'WINDSOR A'),
}

app = App(token=SLACK_BOT_TOKEN)

# ==========================================
# 2. HELPER UTILITIES & WEATHER ENGINE
# ==========================================
def fetch_climate_data(station_id, year):
    """Fetches daily data from Environment Canada API."""
    url = (f'https://climate.weather.gc.ca/climate_data/bulk_data_e.html'
           f'?format=csv&stationID={station_id}&Year={year}'
           f'&Month=1&Day=1&timeframe=2&submit=Download+Data')
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = df.columns.str.strip()
        date_col = [c for c in df.columns if 'Date' in c or 'date' in c][0]
        df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
        return df.dropna(subset=['Date'])
    except Exception as e:
        print(f"Error fetching weather for station {station_id}: {e}")
        return None

def calc_chu_element(tmax, tmin):
    """OMAFRA Corn Heat Unit standard equation."""
    tmax = min(tmax, 30.0)
    ymax = 3.33 * (tmax - 10.0) - 0.084 * (tmax - 10.0)**2 if tmax > 10.0 else 0.0
    ymin = 1.8 * (tmin - 4.44) if tmin > 4.44 else 0.0
    return max(0.0, (ymax + ymin) / 2.0)

def get_field_soil_data(field_name):
    """Loads and returns combined data frame matching the target KML shortname."""
    csv_files = glob.glob(os.path.join(CSV_FOLDER, '*.csv'))
    dfs = []
    for f in csv_files:
        base = os.path.basename(f).replace('.csv', '')
        mapped_name = FIELD_NAME_MAP.get(base, base)
        if mapped_name == field_name:
            df_temp = pd.read_csv(f)
            df_temp['Field'] = field_name
            dfs.append(df_temp)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)

# ==========================================
# 3. SLACK SLASH COMMANDS HANDLERS
# ==========================================

@app.command("/gdd-chu")
def handle_gdd_chu(ack, respond, command):
    """Command: Calculates current vs last year accumulation from planting date."""
    ack()
    field_name = command['text'].strip()
    
    if field_name not in PLANTING_DB:
        respond(f"❌ Field `{field_name}` not recognized or missing a planting date in DB.")
        return
        
    station_info = FIELD_STATION_MAP.get(field_name)
    if not station_info:
        respond(f"❌ No weather station linked to field `{field_name}`.")
        return
        
    station_id, station_name = station_info
    planting_date_str = PLANTING_DB[field_name]
    p_date = pd.to_datetime(planting_date_str)
    current_date = datetime.now()
    
    respond(f"⏳ Fetching climate logs for station {station_name} (ID: {station_id}) from {planting_date_str} to date...")
    
    df_curr = fetch_climate_data(station_id, current_date.year)
    df_prev = fetch_climate_data(station_id, current_date.year - 1)
    
    if df_curr is None or df_prev is None:
        respond("❌ Failed to pull complete reports from Environment Canada servers.")
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
        chu = sub.apply(lambda r: calc_chu_element(r['Tmax'], r['Tmin']) if pd.notna(r['Tmax']) else 0, axis=1).sum()
        return round(gdd, 0), round(chu, 0)

    curr_start = p_date
    curr_end = pd.to_datetime(current_date.strftime('%Y-%m-%d'))
    prev_start = p_date - pd.DateOffset(years=1)
    prev_end = curr_end - pd.DateOffset(years=1)
    
    gdd_c, chu_c = compute_metrics(df_curr, curr_start, curr_end)
    gdd_p, chu_p = compute_metrics(df_prev, prev_start, prev_end)
    
    plt.figure(figsize=(7, 4))
    df_curr_filtered = df_curr[(df_curr['Date'] >= curr_start) & (df_curr['Date'] <= curr_end)].copy()
    if not df_curr_filtered.empty:
        df_curr_filtered['Tmean'] = pd.to_numeric(df_curr_filtered[[c for c in df_curr_filtered.columns if 'Mean Temp' in c][0]], errors='coerce')
        df_curr_filtered['GDD_cum'] = (df_curr_filtered['Tmean'] - GDD_BASE).clip(lower=0).cumsum()
        plt.plot(df_curr_filtered['Date'], df_curr_filtered['GDD_cum'], color='#4a9e6b', linewidth=2.5, label='Current Year')
        
    plt.title(f"Field: {field_name} — In-Season Cumulative GDD Timeline", fontsize=11, fontweight='bold')
    plt.ylabel('Accumulated GDD (°C·days)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=150)
    img_buf.seek(0)
    plt.close()
    
    text_summary = (
        f"🌱 *Field Report: {field_name}* (Planted: {planting_date_str})\n"
        f"───────────────────────────────\n"
        f"📊 *GDD Accumulation (Base 5°C):*\n"
        f"  • Current Season: *{gdd_c:.0f} GDD*\n"
        f"  • Last Season (Same Period): {gdd_p:.0f} GDD ({'ahead' if gdd_c > gdd_p else 'behind'})\n\n"
        f"🌽 *OMAFRA Corn Heat Units (CHU):*\n"
        f"  • Current Season: *{chu_c:.0f} CHU*\n"
        f"  • Last Season (Same Period): {chu_p:.0f} CHU"
    )
    
    app.client.files_upload_v2(
        channel=command['channel_id'],
        initial_comment=text_summary,
        file=img_buf.read(),
        filename=f"{field_name}_weather_timeline.png"
    )

@app.command("/mineralization")
def handle_mineralization(ack, respond, command):
    """Command: Generates sub-field spatial OM nitrogen release map using peer-reviewed equations."""
    ack()
    field_name = command['text'].strip()
    
    df_soil = get_field_soil_data(field_name)
    if df_soil is None:
        respond(f"❌ Could not find high-resolution SoilOptix layers for field `{field_name}` in the soil_data folder.")
        return

    station_info = FIELD_STATION_MAP.get(field_name)
    if not station_info or field_name not in PLANTING_DB:
        respond("❌ Field missing station configuration or target planting data entry.")
        return
        
    df_wx = fetch_climate_data(station_info[0], datetime.now().year)
    mask = (df_wx['Date'] >= pd.to_datetime(PLANTING_DB[field_name])) & (df_wx['Date'] <= pd.to_datetime(datetime.now()))
    sub_wx = df_wx[mask].copy()
    if sub_wx.empty:
        respond(f"❌ Weather data not yet available for {field_name} since planting on {PLANTING_DB[field_name]}.")
        return

    tmean_col = [c for c in sub_wx.columns if 'Mean Temp' in c or 'TMEAN' in c][0]
    sub_wx['Tmean'] = pd.to_numeric(sub_wx[tmean_col], errors='coerce')
    current_gdd = (sub_wx['Tmean'] - GDD_BASE).clip(lower=0).sum()

    df_soil['N_pot'] = df_soil['OM'] * 26.5
    df_soil['N_min_kg_ha'] = df_soil['N_pot'] * (1 - np.exp(-0.0014 * current_gdd))
    
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(df_soil['Longitude'], df_soil['Latitude'], c=df_soil['N_min_kg_ha'], cmap='YlOrRd', s=8, alpha=0.8)
    cbar = plt.colorbar(sc)
    cbar.set_label('Estimated N Released via Mineralization ($kg\ N \cdot ha^{-1}$)', fontsize=10)
    plt.title(f"Field: {field_name} — Organic Matter N Mineralization Map\nCalculated at {current_gdd:.0f} Cumulative GDD", fontsize=11, fontweight='bold')
    plt.xlabel('Longitude', fontsize=9)
    plt.ylabel('Latitude', fontsize=9)
    plt.tight_layout()
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=150)
    img_buf.seek(0)
    plt.close()
    
    avg_release = df_soil['N_min_kg_ha'].mean()
    comment = f"🗺️ *High-Resolution Soil Organic Matter Mineralization Map for {field_name}*\n" \
              f"• Average expected plant-available nitrogen released to date: *{avg_release:.1f} kg N/ha* (~{avg_release*0.89:.1f} lbs/ac).\n" \
              f"• Driven by current heat tracking index ({current_gdd:.0f} GDD) mapped onto localized Gamma-Radiation baseline OM data layers."
              
    app.client.files_upload_v2(
        channel=command['channel_id'],
        initial_comment=comment,
        file=img_buf.read(),
        filename=f"{field_name}_om_mineralization.png"
    )

@app.command("/trial-zones")
def handle_trial_zones(ack, respond, command):
    """Command: Segments soil layers via K-Means and assigns balanced VRN calibration trials."""
    ack()
    args = command['text'].strip().split()
    if not args:
        respond("❌ Please provide parameters. Example: `/trial-zones Wettlaufer 4`")
        return
        
    field_name = args[0]
    n_zones = int(args[1]) if len(args) > 1 else 4
    
    df_soil = get_field_soil_data(field_name)
    if df_soil is None:
        respond(f"❌ Soil layers for `{field_name}` could not be verified.")
        return
        
    SOIL_FEATURES = ['OM', 'Clay', 'Sand', 'pH', 'CEC']
    df_clean = df_soil.dropna(subset=SOIL_FEATURES).copy()
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clean[SOIL_FEATURES])
    
    kmeans = KMeans(n_clusters=n_zones, random_state=42, n_init=10)
    df_clean['Zone'] = kmeans.fit_predict(scaled_data) + 1
    
    trial_rate_mapping = {1: 56, 2: 112, 3: 160, 4: 208, 5: 240}
    df_clean['Trial_Prescription_Rate_kg_ha'] = df_clean['Zone'].map(trial_rate_mapping).fillna(160)
    
    plt.figure(figsize=(8, 6))
    colors_palette = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3']
    for z in sorted(df_clean['Zone'].unique()):
        sub_z = df_clean[df_clean['Zone'] == z]
        plt.scatter(sub_z['Longitude'], sub_z['Latitude'], 
                    label=f"Zone {z} — Target: {trial_rate_mapping.get(z, 160)} kg/ha", 
                    color=colors_palette[(z-1) % 5], s=6, alpha=0.7)
                    
    plt.title(f"K-Means Delineated On-Farm N Trial Zones — Field: {field_name}", fontsize=11, fontweight='bold')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(title='Management Strips', loc='best', fontsize=8)
    plt.tight_layout()
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=150)
    img_buf.seek(0)
    plt.close()
    
    csv_buf = io.StringIO()
    df_clean[['Longitude', 'Latitude', 'Zone', 'Trial_Prescription_Rate_kg_ha']].to_csv(csv_buf, index=False)
    
    app.client.files_upload_v2(
        channel=command['channel_id'],
        file=img_buf.read(),
        filename=f"{field_name}_trial_zones.png",
        initial_comment=f"🤖 *Variable Rate Nitrogen Trial Script Executed successfully for {field_name}*"
    )
    
    app.client.files_upload_v2(
        channel=command['channel_id'],
        content=csv_buf.getvalue(),
        filename=f"{field_name}_prescription_points.csv",
        initial_comment=f"📄 Downstream controller file ready for application monitors (Shape/Points format)."
    )

# ==========================================
# 4. EXECUTION GATEWAY & WEB SERVER HACK
# ==========================================
from flask import Flask
from threading import Thread

# 1. Create a dummy web server
web_app = Flask(__name__)

@web_app.route('/')
def home():
    return "Bot is awake and listening!"

def run_web_server():
    # Render assigns a specific hidden PORT. We must bind to it.
    port = int(os.environ.get("PORT", 8080))
    web_app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    print("🌐 Starting fake web server to trick Render...")
    # Spin the web server off into a background thread
    Thread(target=run_web_server).start()
    
    print("⚡ Starting PRECISION AG BOT via Slack SocketMode...")
    # Run the Slack bot on the main thread
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()