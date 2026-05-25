import pandas as pd
import numpy as np
import glob
import os
import json
import zipfile
import requests
import io
import base64
import math
import xml.etree.ElementTree as ET

# GIS & Routing Libraries
import pyproj
import rasterio
import rasterio.mask
import whitebox

# Headless Matplotlib for thread-safe rendering
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

wbt = whitebox.WhiteboxTools()
wbt.set_verbose_mode(False)

# =========================================================
# 1. CONFIGURATION & TRACKING
# =========================================================

CSV_FOLDER = os.path.dirname(os.path.abspath(__file__))
GIS_CACHE_DIR = os.path.join(CSV_FOLDER, "gis_cache")

if not os.path.exists(GIS_CACHE_DIR):
    os.makedirs(GIS_CACHE_DIR)

TRACKER = {
    'success': set(),
    'missing_kml': set(),
    'gis_failed': set(),
    'csv_failed': set(),
    'duplicates_skipped': set()
}

FILE_NAME_MAP = {
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

KML_ALIAS_MAP = {
    'FieldAndFlock_1': 'FieldAndFlock 1', 'FieldAndFlock_2': 'FieldAndFlock 2',
    'Moosberger_1': 'Moosberger 1', 'Moosberger_2': 'Moosberger 2',
    'Kerrigan': 'Kerrington', 'Peters': 'JD Peters', 'Gerber Acres (1)': 'Gerber Acres',
    'Marvara': 'Marvara / Judd', 'Judd Guevera': 'Marvara / Judd',
    'Brucelea Poultry': 'Brucelea', 'Grubb': 'GerMar Farms (Grubb)', 'GerMar': 'GerMar Farms (Grubb)'
}

# EPSG:4326 (GPS) to EPSG:26917 (UTM Zone 17N - Ontario)
project_to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:26917", always_xy=True)

# =========================================================
# 2. HELPER FUNCTIONS & SANITIZERS
# =========================================================

def safe_float(val):
    if val is None or pd.isna(val): return None
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v): return None
        return v
    except (ValueError, TypeError):
        return None

def resolve_field_name(raw_name):
    if not raw_name: return None
    name = str(raw_name).replace('\\', '/').split('/')[-1]
    
    for ext in ['.csv', '.tif', '.tiff', '.shp', '.kml', '.gpkg', ' (1)', '.zip']:
        name = name.replace(ext, '').replace(ext.upper(), '')
        
    if name in FILE_NAME_MAP: return FILE_NAME_MAP[name]
    if name in FILE_NAME_MAP.values(): return name
    
    for prefix in ['Upside_Robotics_', '_NutrientTexture', '_CSV']:
        name = name.replace(prefix, '')
    name = ' '.join(name.replace('_', ' ').split()).strip()
    
    alias_map = {
        'Clare Horst Home East': 'Clare Horst', 'Moose': 'Moosberger 1', 'Roth': 'Benderbrook 1',
        'Tim60': 'Benderbrook 2', 'Adam Wettlaufer Adam Wettlaufer': 'Wettlaufer',
        'Bercab Bercab 1 Shop': 'Bercab 1', 'Bercab Bercab 2': 'Bercab 2', 'Bercab Farms 1': 'Bercab 1', 'Bercab Farms -2': 'Bercab 2',
        'Biermans Main': 'Biermans', 'Biermans Farms HM Limited - 2 systems': 'Biermans',
        'Brad Haack Schause': 'Schaus', 'Brad Haack': 'Schaus',
        'Brucelea Brucelea for Upside': 'Brucelea', 'Brucelea Poultry': 'Brucelea', 'Brucelea Poultry (2)': 'Brucelea',
        'Christie Christies 2': 'Christie-2', 'Christie 2': 'Christie-2',
        'Christie Christie 1': 'Christie-1', 'Christie 1': 'Christie-1', 'Christie -1': 'Christie-1',
        'Ed Burm Ed Burm 1 and 2': 'Burm', 'Burm East': 'Burm', 'Burm West': 'Burm', 'Burm 1': 'Burm', 'Burm 2': 'Burm',
        'Field and Flock Demaree': 'FieldAndFlock 1', 'Field and Flock Farms (1)': 'FieldAndFlock 1', 'FieldAndFlock 1': 'FieldAndFlock 1',
        'Field and Flock DeVries': 'FieldAndFlock 2', 'Field and Flock (3)': 'FieldAndFlock 2', 'FieldAndFlock 2': 'FieldAndFlock 2',
        'Gerard Grubb Gerard Grubb': 'GerMar Farms (Grubb)', 'Grubb': 'GerMar Farms (Grubb)', 'GerMar': 'GerMar Farms (Grubb)', 'Grubb (3)': 'GerMar Farms (Grubb)',
        'Gerber Acres Gerber 1': 'Gerber Acres', 'Gerber Acres (1)': 'Gerber Acres',
        'Gerrits Gerrits': 'Gerrits', 'Martin Gerrits-1': 'Gerrits', 'Martin Gerrits': 'Gerrits',
        'Greg Leis Tracks': 'Leis', 'Greg Leis': 'Leis',
        'Highland Farms Highland 1': 'Highland', 'Highland Farms': 'Highland',
        'JD Peters JD Peters 13th Concession': 'JD Peters', 'Peters': 'JD Peters', 'Peters (1)': 'JD Peters',
        'Kerrington Kerrington': 'Kerrington', 'Kerrigan': 'Kerrington', 'Mike Kerrigan': 'Kerrington',
        'Klavans Klavan 7440': 'Klavan', 'Klavan 2': 'Klavan', 'Klavan-1': 'Klavan', 'Klavan-2': 'Klavan',
        'Langs main': 'Lang', 'Lang farms': 'Lang',
        'Marvara Marvara 1': 'Marvara / Judd', 'Judd Guevera': 'Marvara / Judd', 'Marvara': 'Marvara / Judd',
        'Renwick Renwick 1': 'Renwick 1', 'Renwick-1': 'Renwick 1',
        'Renwick Renwick 2': 'Renwick 2', 'Renwick-2': 'Renwick 2',
        'Roland McAlpine McAlpine 1': 'McAlpine', 'Roland McAlpine McAlpine 2': 'McAlpine', 'Roland McAlpine': 'McAlpine',
        'Russ Schumm Schumm 401': 'Schumhaven', 'Schumhaven Farms': 'Schumhaven',
        'Scott Campbell Campbell Home': 'Campbell', 'Scott Campbell': 'Campbell',
        'Sydenham Sydenham 1': 'Sydenham 1', 'Sydenham-1': 'Sydenham 1',
        'Sydenham Sydenham 2 North': 'Sydenham 2 North', 'Sydenham 2 North': 'Sydenham 2 North',
        'Sydenham Sydenham 2 South': 'Sydenham 2 South', 'Sydenham 2 South': 'Sydenham 2 South', 'Sydenham-2': 'Sydenham 2 South',
        'Triaro Triaro 18 Line': 'Triaro', 'Triaro Farms Inc': 'Triaro',
        'Triple Lane Farm Triple Lane 179 Howell': 'Triple Lane Farms', 'Triple Lane Farms (2)': 'Triple Lane Farms', 'Triple Lane': 'Triple Lane Farms',
        'Veldale Veldale Research 74': 'Veldale', 'Weckers': 'Wecker'
    }
    if name in alias_map: return alias_map[name]
    
    valid_targets = set(FILE_NAME_MAP.values()).union(set(alias_map.values()))
    for target in sorted(valid_targets, key=len, reverse=True):
        if target.lower() in name.lower():
            return target
    return name

def get_texture_drainage_modifier(texture):
    """Agronomic permeability shift based on soil texture."""
    modifiers = {
        'Sand': -1.5, 'Loamy Sand': -1.2, 'Sandy Loam': -0.8,
        'Loam': 0.0, 'Silt Loam': 0.4, 'Silt': 0.7,
        'Clay Loam': 1.0, 'Silty Clay Loam': 1.4, 'Sandy Clay Loam': 1.2,
        'Clay': 2.0, 'Silty Clay': 2.2, 'Sandy Clay': 1.8
    }
    return modifiers.get(texture, 0.0)

def get_twi_label(adjusted_twi):
    """Calibrated TWI thresholds (using p90 + texture modifiers)."""
    if adjusted_twi is None: return "Unknown"
    if adjusted_twi < 8.5: return "Rapid Shedding"
    if adjusted_twi < 10.0: return "Well Drained"
    if adjusted_twi < 11.5: return "Moderate Drainage"
    if adjusted_twi < 13.0: return "High Moisture Retention"
    if adjusted_twi < 14.5: return "Slow Drainage / Often Wet"
    return "Prone to Pooling"

def get_col_stats(df, possible_names):
    for col in df.columns:
        if col.strip().lower() in [n.lower() for n in possible_names]:
            s = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(s) > 0: 
                return safe_float(s.mean()), safe_float(s.min()), safe_float(s.max())
    return None, None, None

def get_val_from_row(row, possible_names):
    for col in row.index:
        if str(col).strip().lower() in [n.lower() for n in possible_names]:
            return row[col]
    return None

def classify_texture(clay, silt, sand):
    if clay is None or silt is None or sand is None: return 'N/A'
    if clay >= 40: return 'Sandy Clay' if sand >= 45 else 'Silty Clay' if silt >= 40 else 'Clay'
    if 27 <= clay < 40: return 'Sandy Clay Loam' if sand >= 45 else 'Clay Loam' if 20 <= silt <= 45 and sand < 45 else 'Silty Clay Loam'
    if 20 <= clay < 27: return 'Sandy Loam' if sand >= 52 and silt < 28 else 'Loam' if 28 <= silt and sand < 52 else 'Clay Loam'
    if silt >= 80: return 'Silt'
    if silt >= 50: return 'Silt Loam'
    if sand >= 85: return 'Sand'
    if sand >= 70: return 'Loamy Sand'
    return 'Sandy Loam' if sand >= 43 else 'Loam'

# =========================================================
# 3. KML POLYGON PARSER (UTM ZONE 17N)
# =========================================================

kml_boundaries = {}
kml_files = glob.glob(os.path.join(CSV_FOLDER, "*.kml"))

if kml_files:
    print(f"\n[+] Found KML boundary file: {os.path.basename(kml_files[0])}")
    try:
        tree = ET.parse(kml_files[0])
        root = tree.getroot()
        for elem in root.iter():
            if '}' in elem.tag:
                elem.tag = elem.tag.split('}', 1)[1]
        for placemark in root.findall('.//Placemark'):
            name_elem = placemark.find('name')
            coords_elem = placemark.find('.//Polygon/outerBoundaryIs/LinearRing/coordinates')
            
            if name_elem is not None and coords_elem is not None:
                raw_name = name_elem.text.strip()
                if raw_name in KML_ALIAS_MAP:
                    field_name = KML_ALIAS_MAP[raw_name]
                else:
                    field_name = resolve_field_name(raw_name)
                
                if field_name:
                    coords = []
                    for pt_str in coords_elem.text.strip().split():
                        parts = pt_str.split(',')
                        if len(parts) >= 2:
                            lon = safe_float(parts[0])
                            lat = safe_float(parts[1])
                            if lon is not None and lat is not None:
                                # Convert GPS to UTM Zone 17N
                                x, y = project_to_utm.transform(lon, lat)
                                coords.append((x, y))
                                
                    if len(coords) >= 3:
                        if coords[0] != coords[-1]:
                            coords.append(coords[0])
                        kml_boundaries[field_name] = {'type': 'Polygon', 'coordinates': [coords]}
        print(f"  ✓ Extracted precision UTM boundaries for {len(kml_boundaries)} fields.")
    except Exception as e:
        print(f"  X Failed to parse KML: {e}")

# =========================================================
# 4. PRE-LOAD TERRAIN METRICS
# =========================================================

terrain_data_map = {}
terrain_search = glob.glob(os.path.join(CSV_FOLDER, "**/field_terrain_metrics.csv"), recursive=True)

if terrain_search:
    terrain_csv_path = terrain_search[0]
    try:
        tdf = pd.read_csv(terrain_csv_path)
        name_col = next((c for c in tdf.columns if 'field' in c.lower() or 'name' in c.lower()), None)
        if not name_col: name_col = tdf.columns[0] 
        
        if name_col:
            for _, row in tdf.iterrows():
                raw_name = str(row[name_col])
                field_name = resolve_field_name(raw_name)
                if field_name:
                    terrain_data_map[field_name] = {
                        'tri': safe_float(get_val_from_row(row, ['tri', 'ruggedness'])),
                        'topo_mod': safe_float(get_val_from_row(row, ['mineralization_topo_modifier', 'topo_modifier'])),
                    }
    except Exception:
        pass

# =========================================================
# 5. ONTARIO LIO WCS & WHITEBOX PIPELINE (UTM 26917)
# =========================================================

def process_field_gis(field_name, min_lon, min_lat, max_lon, max_lat):
    # Buffer in degrees before converting to UTM
    buf = 0.0015
    minx_geo, miny_geo = min_lon - buf, min_lat - buf
    maxx_geo, maxy_geo = max_lon + buf, max_lat + buf
    
    # Convert to UTM
    minx, miny = project_to_utm.transform(minx_geo, miny_geo)
    maxx, maxy = project_to_utm.transform(maxx_geo, maxy_geo)
    
    # Dynamically calculate image size (Targeting ~2m resolution)
    width = int(maxx - minx) // 2
    height = int(maxy - miny) // 2
    width = max(400, min(1800, width))
    height = max(400, min(1800, height))

    safe_name = field_name.replace(' ', '_').replace('/', '_')
    # Using v3 to bust old Web Mercator caches
    dem_path = os.path.join(GIS_CACHE_DIR, f"{safe_name}_dem_v3.tif")
    breached_path = os.path.join(GIS_CACHE_DIR, f"{safe_name}_breach_v3.tif")
    twi_path = os.path.join(GIS_CACHE_DIR, f"{safe_name}_twi_v3.tif")
    scca_path = os.path.join(GIS_CACHE_DIR, f"{safe_name}_scca_v3.tif")
    slope_path = os.path.join(GIS_CACHE_DIR, f"{safe_name}_slope_v3.tif")

    poly_geom = kml_boundaries.get(field_name)
    if not poly_geom:
        TRACKER['missing_kml'].add(field_name)

    results = {
        'dem_b64': None, 'twi_b64': None,
        'twi_p90': None, 'slope_p75': None,
        'topo_desc': 'Unknown',
        'elev_min': None, 'elev_max': None, 'elev_var': None
    }

    try:
        if not os.path.exists(dem_path):
            wcs_url = (f"https://ws.geoservices.lrc.gov.on.ca/arcgis5/rest/services/Elevation/"
                       f"Ontario_DTM_LidarDerived/ImageServer/exportImage?"
                       f"bbox={minx},{miny},{maxx},{maxy}&bboxSR=26917&imageSR=26917"
                       f"&size={width},{height}&format=tiff&pixelType=F32&noDataInterpretation=esriNoDataMatchAny&f=image")
            
            r = requests.get(wcs_url, timeout=45)
            if (r.status_code == 200 and len(r.content) > 10000 and r.content[:2] in [b'II', b'MM']):
                with open(dem_path, 'wb') as f: f.write(r.content)
            else:
                TRACKER['gis_failed'].add(field_name)
                return results

        if not os.path.exists(twi_path):
            wbt.breach_depressions(dem_path, breached_path)
            # CRITICAL FIX: Explicitly setting units="percent"
            wbt.slope(breached_path, slope_path, units="percent")
            wbt.d_inf_flow_accumulation(breached_path, scca_path, out_type="Specific Catchment Area")
            try:
                wbt.wetness_index(scca_path, slope_path, twi_path)
            except Exception:
                pass

        # Mask & Process DEM
        with rasterio.open(dem_path) as src:
            if poly_geom:
                try:
                    dem_data, _ = rasterio.mask.mask(src, [poly_geom], crop=True, nodata=np.nan)
                    dem_data = dem_data[0]
                except Exception as e:
                    print(f"    [!] DEM KML Mask Failed for {field_name}: {e}")
                    dem_data = src.read(1)
            else:
                dem_data = src.read(1)
                
            dem_data = dem_data.astype(np.float32)
            dem_data[dem_data <= -9999] = np.nan 
            
            valid_dem = dem_data[~np.isnan(dem_data)]
            if valid_dem.size > 0:
                results['elev_min'] = safe_float(np.min(valid_dem))
                results['elev_max'] = safe_float(np.max(valid_dem))
                if results['elev_max'] is not None and results['elev_min'] is not None:
                    results['elev_var'] = results['elev_max'] - results['elev_min']
        
        # Mask & Process Slope (Percentiles)
        with rasterio.open(slope_path) as src:
            if poly_geom:
                try:
                    slope_data, _ = rasterio.mask.mask(src, [poly_geom], crop=True, nodata=np.nan)
                    slope_data = slope_data[0]
                except Exception:
                    slope_data = src.read(1)
            else:
                slope_data = src.read(1)
                
            slope_data = slope_data.astype(np.float32)
            slope_data[slope_data < 0] = np.nan 
            
            valid_slope = slope_data[~np.isnan(slope_data)]
            if valid_slope.size > 0:
                slope_p75 = safe_float(np.percentile(valid_slope, 75))
                results['slope_p75'] = slope_p75
                if slope_p75 is not None:
                    # CRITICAL FIX: Calibrated thresholds for % slope
                    if slope_p75 >= 8.0: results['topo_desc'] = "Hilly / Sloped"
                    elif slope_p75 >= 3.0: results['topo_desc'] = "Gently Rolling"
                    elif slope_p75 >= 1.0: results['topo_desc'] = "Undulating"
                    else: results['topo_desc'] = "Flat"

        # Mask & Process TWI (Percentiles)
        if os.path.exists(twi_path):
            with rasterio.open(twi_path) as src:
                if poly_geom:
                    try:
                        twi_data, _ = rasterio.mask.mask(src, [poly_geom], crop=True, nodata=np.nan)
                        twi_data = twi_data[0]
                    except Exception:
                        twi_data = src.read(1)
                else:
                    twi_data = src.read(1)
                
                twi_data = twi_data.astype(np.float32)
                twi_data[twi_data <= 0] = np.nan
                
                valid_twi = twi_data[~np.isnan(twi_data)]
                if valid_twi.size > 0:
                    results['twi_p90'] = safe_float(np.percentile(valid_twi, 90))
                
            # Draw Maps
            fig = Figure(figsize=(4, 3))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            im1 = ax.imshow(dem_data, cmap='terrain')
            fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04, label="Elevation (m)")
            ax.axis('off')
            fig.tight_layout()
            buf_img = io.BytesIO()
            canvas.print_png(buf_img)
            results['dem_b64'] = base64.b64encode(buf_img.getvalue()).decode('utf-8')

            fig2 = Figure(figsize=(4, 3))
            canvas2 = FigureCanvas(fig2)
            ax2 = fig2.add_subplot(111)
            im2 = ax2.imshow(twi_data, cmap='Blues')
            fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="TWI (Moisture)")
            ax2.axis('off')
            fig2.tight_layout()
            buf_img2 = io.BytesIO()
            canvas2.print_png(buf_img2)
            results['twi_b64'] = base64.b64encode(buf_img2.getvalue()).decode('utf-8')

    except Exception as e:
        print(f"\nGIS FAILURE [{field_name}]: {e}")
        TRACKER['gis_failed'].add(field_name)

    return results

# =========================================================
# 6. PARSE SOIL SAMPLES & LINK DATA
# =========================================================

processed_fields = {}

def process_dataframe(df, filename, field_name):
    if field_name in processed_fields:
        TRACKER['duplicates_skipped'].add(field_name)
        return

    try:
        clay, _, _ = get_col_stats(df, ['clay', 'clay %', 'clay_pct'])
        sand, _, _ = get_col_stats(df, ['sand', 'sand %', 'sand_pct'])
        silt, _, _ = get_col_stats(df, ['silt', 'silt %', 'silt_pct'])
        om, _, _   = get_col_stats(df, ['om', 'organic matter', 'om %'])
        ph, _, _   = get_col_stats(df, ['ph', 'soil ph'])
        
        n, _, _    = get_col_stats(df, [
            'n', 'nitrogen', 'total n', 'no3', 'n (ppm)', 
            'no3-n', 'no3n', 'nitrate', 'nitrate-n', 'nitrogen ppm', 'nitrate-n ppm'
        ])
        k, _, _    = get_col_stats(df, ['k', 'potassium', 'k (ppm)'])
        texture = classify_texture(clay, silt, sand)
        
        n_lbs_ac = (n * 2) if n is not None else None

        t_data = terrain_data_map.get(field_name, {})
        
        lat_mean, lat_min, lat_max = get_col_stats(df, ['lat', 'latitude', 'y'])
        lon_mean, lon_min, lon_max = get_col_stats(df, ['lon', 'longitude', 'x'])
        
        gis_data = {}
        twi_desc = "Unknown"
        
        if (lat_min is not None and lon_min is not None and lat_max is not None and lon_max is not None):
            gis_data = process_field_gis(field_name, lon_min, lat_min, lon_max, lat_max)
            
            # Apply Texture Drainage Modifier
            raw_twi_p90 = gis_data.get('twi_p90')
            if raw_twi_p90 is not None:
                texture_mod = get_texture_drainage_modifier(texture)
                adjusted_twi = raw_twi_p90 + texture_mod
                twi_desc = get_twi_label(adjusted_twi)
        
        processed_fields[field_name] = {
            "summary": {
                "clay": clay, "sand": sand, "silt": silt, "om": om,
                "pH": ph, "N": n, "N_lbs_ac": n_lbs_ac, "K": k, 
                "twi_desc": twi_desc,
                "elev_var": gis_data.get('elev_var'), 
                "elev_min": gis_data.get('elev_min'), 
                "elev_max": gis_data.get('elev_max'),
                "topo_desc": gis_data.get('topo_desc', 'Unknown'), 
                "tri": t_data.get('tri'), 
                "topo_mod": t_data.get('topo_mod'),
                "dem_b64": gis_data.get('dem_b64'), 
                "twi_b64": gis_data.get('twi_b64'),
                "texture": texture, "n_samples": len(df)
            }
        }
        
        TRACKER['success'].add(field_name)
        print(f"  ✓ Connected Profile: {field_name} ({len(df)} points)")
        
    except Exception as e:
        print(f"\nCSV FAILURE [{field_name}]: {e}")
        TRACKER['csv_failed'].add(field_name)

print(f"\nSearching deep inside directory: {CSV_FOLDER}")

def safe_read_csv(filepath):
    try:
        return pd.read_csv(filepath, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(filepath, encoding='latin1', low_memory=False)

csv_files = glob.glob(os.path.join(CSV_FOLDER, "**/*.csv"), recursive=True)
for filepath in csv_files:
    filename = os.path.basename(filepath)
    if "weekly" in filename or "daily" in filename or "metrics" in filename or "report" in filename or "terrain" in filename: continue
    field_name = resolve_field_name(filename)
    if not field_name: continue
    df = safe_read_csv(filepath)
    process_dataframe(df, filename, field_name)

zip_files = glob.glob(os.path.join(CSV_FOLDER, "**/*.zip"), recursive=True)
if zip_files:
    print(f"\n[+] Found {len(zip_files)} compressed batches recursively. Peering inside...")

for zip_path in zip_files:
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            for file_info in z.infolist():
                if file_info.filename.endswith('.csv'):
                    filename = os.path.basename(file_info.filename)
                    if "weekly" in filename or "daily" in filename or "metrics" in filename or "terrain" in filename: continue
                    field_name = resolve_field_name(filename)
                    if not field_name: continue
                    with z.open(file_info) as f:
                        df = pd.read_csv(f, low_memory=False)
                        process_dataframe(df, filename, field_name)
    except Exception as e:
        pass

json_data = json.dumps(processed_fields).replace("</script>", "<\\/script>")

# =========================================================
# 7. GENERATE CLEAN WEB REPORT
# =========================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Upside Robotics · Field Intelligence</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
  :root{
    --bg:#f8fafc; --bg1:#ffffff; --bg2:#f1f5f9; --bg3:#e2e8f0;
    --border:#cbd5e1; --border2:#94a3b8; --txt:#0f172a; --txt2:#475569; --txt3:#64748b;
    --green:#10b981; --green-d:#047857; --amber:#f59e0b; --amber-d:#b45309;
    --red:#ef4444; --blue:#3b82f6; --blue-d:#1d4ed8; --teal:#14b8a6;
    --purple:#8b5cf6; --accent:#10b981; --r:8px; --r2:14px;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{ background:var(--bg); color:var(--txt); font-family:'Syne',sans-serif; min-height:100vh; overflow-x:hidden; }
  
  .topbar{ display:flex;align-items:center;justify-content:space-between; padding:14px 28px; background:rgba(255,255,255,.92); border-bottom:1px solid var(--border); backdrop-filter:blur(12px); position:sticky;top:0;z-index:100; }
  .logo{ display:flex;align-items:center;gap:10px; font-weight:800;font-size:1.15rem;letter-spacing:-.02em; }
  .logo-icon{ width:32px;height:32px; background:linear-gradient(135deg,var(--green),var(--teal)); border-radius:8px; display:flex;align-items:center;justify-content:center; font-size:.85rem; color:white; }
  .topbar-right{display:flex;align-items:center;gap:12px}
  .badge{ padding:4px 10px;border-radius:20px; font-family:'JetBrains Mono',monospace; font-size:.72rem;font-weight:600; background:var(--bg3);border:1px solid var(--border2); color:var(--txt2); }
  .badge.live{border-color:var(--green-d);color:var(--green-d);background:rgba(16,185,129,.1)}
  .btn{ padding:7px 16px;border-radius:var(--r); font-family:'Syne',sans-serif;font-size:.82rem;font-weight:700; cursor:pointer;border:1px solid var(--border2); background:var(--bg1);color:var(--txt); transition:all .15s; }
  .btn:hover{background:var(--bg2);border-color:var(--txt)}
  .btn.primary{ background:var(--txt);color:#fff;border-color:transparent; }

  .hero{ padding:36px 28px 20px; display:flex;align-items:flex-end;justify-content:space-between; flex-wrap:wrap;gap:20px; }
  .hero h1{ font-size:clamp(1.6rem,3vw,2.4rem); font-weight:800;letter-spacing:-.04em; line-height:1.1; }
  .hero h1 span{color:var(--green-d)}
  .hero-sub{color:var(--txt2);font-size:.9rem;margin-top:6px;font-weight:500}

  .controls{ padding:0 28px 20px; display:flex;align-items:center;gap:12px;flex-wrap:wrap; }
  .search-wrap{ position:relative;flex:1;min-width:200px;max-width:320px; }
  .search-wrap input{ width:100%;padding:9px 12px 9px 36px; background:var(--bg1);border:1px solid var(--border2); border-radius:var(--r);color:var(--txt); font-family:'Syne',sans-serif;font-size:.85rem; transition:border-color .15s; }
  .search-wrap input:focus{outline:none;border-color:var(--green)}
  .search-wrap::before{ content:'⌕';position:absolute;left:10px;top:50%;transform:translateY(-50%); color:var(--txt3);font-size:1.1rem;pointer-events:none; }
  select{ padding:9px 12px;background:var(--bg1);border:1px solid var(--border2); border-radius:var(--r);color:var(--txt); font-family:'Syne',sans-serif;font-size:.85rem;cursor:pointer; transition:border-color .15s; font-weight:500; }

  .stats-row{ padding:0 28px 24px; display:flex;gap:12px;flex-wrap:wrap; }
  .stat-card{ background:var(--bg1);border:1px solid var(--border); border-radius:var(--r);padding:14px 18px; min-width:130px;flex:1; display:flex;flex-direction:column;gap:4px; box-shadow:0 1px 3px rgba(0,0,0,0.05); }
  .stat-card .val{ font-size:1.6rem;font-weight:800;letter-spacing:-.04em; color:var(--txt); }
  .stat-card .val.green{color:var(--green-d)}
  .stat-card .val.blue{color:var(--blue-d)}
  .stat-card .lbl{color:var(--txt3);font-size:.75rem;font-weight:700;letter-spacing:.05em;text-transform:uppercase}

  .groups{padding:0 28px 60px;display:flex;flex-direction:column;gap:28px}
  .group-header{ display:flex;align-items:center;gap:14px; margin-bottom:14px; }
  .group-name{ font-size:1.1rem;font-weight:800;letter-spacing:-.02em; }
  .group-count{ padding:3px 9px;border-radius:12px; background:var(--bg3); font-family:'JetBrains Mono',monospace; font-size:.75rem;color:var(--txt2);font-weight:600; }
  .group-line{flex:1;height:1px;background:var(--border)}
  .fields-grid{ display:grid; grid-template-columns:repeat(auto-fill,minmax(310px,1fr)); gap:12px; }

  .field-card{ background:var(--bg1); border:1px solid var(--border); border-radius:var(--r2); padding:16px; cursor:pointer; transition:all .2s; position:relative; overflow:hidden; box-shadow:0 1px 3px rgba(0,0,0,0.02); }
  .field-card:hover{ border-color:var(--border2); transform:translateY(-2px); box-shadow:0 4px 12px rgba(0,0,0,0.05); }
  .field-card.selected{ border-color:var(--green); background:rgba(16,185,129,.03); border-width:2px; padding:15px; }
  .field-card.no-data{ opacity:.6; background:var(--bg2); cursor:not-allowed; }
  
  .card-top{ display:flex;align-items:flex-start;justify-content:space-between; margin-bottom:12px; }
  .field-name{ font-weight:800;font-size:1.05rem;letter-spacing:-.02em; line-height:1.2; color:var(--txt); }
  .planted-badge{ font-family:'JetBrains Mono',monospace; font-weight:600; font-size:.68rem;padding:3px 7px; border-radius:4px; white-space:nowrap; }
  .planted-badge.planted{background:rgba(16,185,129,.15);color:var(--green-d)}
  .planted-badge.not-planted{background:var(--bg3);color:var(--txt3)}

  .card-metrics{ display:grid;grid-template-columns:1fr 1fr;gap:8px; margin-bottom:10px; }
  .metric{ background:var(--bg);border-radius:var(--r); padding:8px 10px; border:1px solid var(--border); }
  .metric.full{grid-column:span 2}
  .metric-lbl{ font-size:.65rem;font-weight:700;letter-spacing:.06em; text-transform:uppercase;color:var(--txt3);margin-bottom:3px; }
  .metric-val{ font-family:'JetBrains Mono',monospace; font-size:.85rem;font-weight:600;color:var(--txt); }
  .metric-val.green{color:var(--green-d)} .metric-val.amber{color:var(--amber-d)} .metric-val.red{color:var(--red)} .metric-val.blue{color:var(--blue-d)}
  .texture-pill{ display:inline-block;padding:2px 8px;border-radius:10px; font-size:.7rem;font-weight:700;letter-spacing:.03em; }
  
  .detail-overlay{ position:fixed;inset:0; background:rgba(0,0,0,.4); z-index:200; backdrop-filter:blur(2px); opacity:0;pointer-events:none; transition:opacity .25s; }
  .detail-overlay.open{opacity:1;pointer-events:all}
  .detail-panel{ position:fixed;right:0;top:0;bottom:0; width:min(540px,95vw); background:var(--bg1); border-left:1px solid var(--border); overflow-y:auto; transform:translateX(100%); transition:transform .28s cubic-bezier(.4,0,.2,1); z-index:201; box-shadow:-5px 0 25px rgba(0,0,0,0.1); }
  .detail-panel.open{transform:translateX(0)}
  .detail-header{ padding:20px 24px 16px; border-bottom:1px solid var(--border); display:flex;align-items:flex-start;justify-content:space-between; position:sticky;top:0;background:var(--bg1);z-index:1; }
  .detail-title{font-size:1.4rem;font-weight:800;letter-spacing:-.03em}
  .detail-sub{color:var(--txt2);font-size:.85rem;margin-top:3px;font-weight:500;}
  .close-btn{ background:var(--bg);border:1px solid var(--border); border-radius:50%;width:32px;height:32px; cursor:pointer;color:var(--txt); display:flex;align-items:center;justify-content:center; font-size:1rem;transition:background .15s;flex-shrink:0; font-weight:bold; }
  .close-btn:hover{background:var(--bg3)}
  .detail-body{padding:20px 24px}

  .d-section{margin-bottom:24px}
  .d-section-title{ font-size:.8rem;font-weight:800;letter-spacing:.08em; text-transform:uppercase;color:var(--txt); margin-bottom:12px;padding-bottom:6px; border-bottom:2px solid var(--border); }
  .d-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  .d-stat{ background:var(--bg);border:1px solid var(--border); border-radius:var(--r); padding:12px 14px; }
  .d-stat .lbl{color:var(--txt3);font-size:.7rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;margin-bottom:4px}
  .d-stat .val{font-family:'JetBrains Mono',monospace;font-size:1.05rem;font-weight:600}
  .d-stat .sub{color:var(--txt2);font-size:.72rem;margin-top:3px;font-weight:500}
  
  .map-tabs { display:flex; gap:8px; margin-bottom: 12px; }
  .map-tab { flex:1; padding:8px; background:var(--bg2); border:1px solid var(--border); border-radius:var(--r); font-weight:700; font-size:0.8rem; color:var(--txt2); cursor:pointer; transition:all 0.2s;}
  .map-tab.active { background:var(--blue); color:white; border-color:var(--blue-d);}
  .map-image-container { width:100%; border-radius:var(--r); background:var(--bg2); border:1px solid var(--border); padding:4px; display:flex; justify-content:center;}
  .map-image { max-width:100%; height:auto; border-radius:var(--r); display:none; }
  .map-image.active { display:block; }
</style>
</head>
<body>
<div id="app">
<nav class="topbar">
  <div class="logo"><div class="logo-icon">🌱</div><span>Upside Robotics</span></div>
  <div class="topbar-right"><span class="badge live">LIO Lidar Connected</span><span class="badge" id="date-badge"></span></div>
</nav>

<div id="main-content">
<div class="hero">
  <div><h1>Field <span>Intelligence</span><br>Dashboard</h1><p class="hero-sub">Soil Data mapped to Whitebox Hydrology & LIO LiDAR</p></div>
  <div style="display:flex;gap:10px">
    <button class="btn" onclick="toggleAllGroups()">Toggle Groups</button>
    <button class="btn primary" onclick="exportSelection()">Export Selected</button>
  </div>
</div>

<div class="controls">
  <div class="search-wrap"><input type="text" id="search" placeholder="Search fields…" oninput="filterFields()"/></div>
  <select id="sort-select" onchange="sortFields()">
    <option value="name">Sort: Name</option><option value="om_desc">Sort: OM High→Low</option>
    <option value="clay_desc">Sort: Clay %</option><option value="n_desc">Sort: Nitrogen High→Low</option>
  </select>
  <select id="filter-planted" onchange="filterFields()"><option value="all">All Fields</option><option value="planted">Planted Only</option><option value="not-planted">Not Planted</option></select>
</div>

<div class="stats-row" id="stats-row">
  <div class="stat-card"><div class="val green" id="stat-fields">0</div><div class="lbl">Total Fields</div></div>
  <div class="stat-card"><div class="val amber" id="stat-planted">0</div><div class="lbl">Planted</div></div>
  <div class="stat-card"><div class="val blue" id="stat-avg-om">—</div><div class="lbl">Avg OM %</div></div>
</div>

<div class="groups" id="groups-container"></div>
</div>

<div class="detail-overlay" id="detail-overlay" onclick="closeDetail()"></div>
<div class="detail-panel" id="detail-panel">
  <div class="detail-header">
    <div><div class="detail-title" id="dp-title">Field Name</div><div class="detail-sub" id="dp-sub">Group</div></div>
    <button class="close-btn" onclick="closeDetail()">✕</button>
  </div>
  <div class="detail-body" id="dp-body"></div>
</div>
</div>

<script id="field-data" type="application/json">
__JSON_DATA_HERE__
</script>

<script>
let fieldData = {};
try {
    const rawData = document.getElementById('field-data').textContent;
    fieldData = JSON.parse(rawData);
} catch (e) {
    console.error("CRITICAL: Failed to parse Python data injection.", e);
    alert("Data Parsing Error. Open your browser console (F12) for details.");
}

const FIELD_GROUPS = {
  'South': ['Gerrits', 'Kerrington', 'Sydenham 1', 'Sydenham 2 North', 'Sydenham 2 South', 'Burm', 'Campbell', 'Wecker', 'Bercab 1', 'Bercab 2'],
  'South Central': ['McAlpine', 'FieldAndFlock 1', 'FieldAndFlock 2', 'Moosberger 1', 'Moosberger 2', 'JD Peters', 'Veldale'],
  'Central': ['Benderbrook 1', 'Benderbrook 2', 'Schumhaven', 'Triple Lane Farms', 'Harrison Farms', 'Clare Horst', 'Gerber Acres', 'Leis'],
  'Arthur': ['Klavan', 'Triaro', 'Marvara / Judd'],
  'Mildmay': ['GerMar Farms (Grubb)', 'Schaus', 'Lang', 'Renwick 1', 'Renwick 2'],
  'West Coast': ['Wettlaufer', 'Brucelea'],
  'North': ['Biermans', 'Christie-1', 'Christie-2', 'Highland']
};

const PLANTING_DB = {
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
    'Bercab 1': null, 'Bercab 2': null, 'Sydenham 1': null, 'Sydenham 2 North': null, 
    'Sydenham 2 South': null, 'McAlpine': null, 'FieldAndFlock 2': null, 'Moosberger 1': null, 
    'Clare Horst': null, 'Marvara / Judd': null, 'Biermans': null
};

const TEXTURE_COLORS = { 'Clay':'#ec4899','Silty Clay':'#8b5cf6','Sandy Clay':'#f97316','Clay Loam':'#d946ef', 'Silty Clay Loam':'#a855f7','Sandy Clay Loam':'#f97316','Loam':'#10b981', 'Sandy Loam':'#f59e0b','Silt Loam':'#3b82f6','Silt':'#6366f1','Loamy Sand':'#eab308','Sand':'#facc15'};
const TEXTURE_BG = { 'Clay':'rgba(236,72,153,.1)','Silty Clay':'rgba(139,92,246,.1)','Sandy Clay':'rgba(249,115,22,.1)', 'Clay Loam':'rgba(217,70,239,.1)','Silty Clay Loam':'rgba(168,85,247,.1)','Sandy Clay Loam':'rgba(249,115,22,.1)', 'Loam':'rgba(16,185,129,.1)','Sandy Loam':'rgba(245,158,11,.1)','Silt Loam':'rgba(59,130,246,.1)', 'Silt':'rgba(99,102,241,.1)','Loamy Sand':'rgba(234,179,8,.1)','Sand':'rgba(250,204,21,.1)'};

let selectedFields = new Set();
let collapsedGroups = new Set();

function getFieldSummary(fieldName) { return fieldData[fieldName]?.summary || null; }

function updateStats() {
  const allParsed = Object.keys(fieldData);
  document.getElementById('stat-fields').textContent = allParsed.length;
  document.getElementById('stat-planted').textContent = allParsed.filter(f=>PLANTING_DB[f]).length;
  const oms = allParsed.map(f=>getFieldSummary(f)?.om).filter(v=>v!==undefined && v!==null);
  document.getElementById('stat-avg-om').textContent = oms.length ? (oms.reduce((a,b)=>a+b)/oms.length).toFixed(1)+'%' : '—';
}

function fieldMatchesFilter(fieldName, filters) {
  if (filters.search && !fieldName.toLowerCase().includes(filters.search)) return false;
  if (filters.planted === 'planted' && !PLANTING_DB[fieldName]) return false;
  if (filters.planted === 'not-planted' && PLANTING_DB[fieldName]) return false;
  return true;
}

function renderGroups() {
  const filters = { search: document.getElementById('search').value.toLowerCase(), planted: document.getElementById('filter-planted').value, sort: document.getElementById('sort-select').value };
  const container = document.getElementById('groups-container'); container.innerHTML = '';
  const allParsedFields = Object.keys(fieldData);
  
  for (const [groupName, baseFields] of Object.entries(FIELD_GROUPS)) {
    let fields = baseFields.filter(f => allParsedFields.includes(f));
    if (groupName === 'North') fields = fields.concat(allParsedFields.filter(f => !Object.values(FIELD_GROUPS).flat().includes(f)));
      
    const filtered = fields.filter(f=>fieldMatchesFilter(f, filters)).sort((a,b) => {
      const sa = getFieldSummary(a) || {}, sb = getFieldSummary(b) || {};
      switch(filters.sort) {
        case 'om_desc': return (sb.om||0)-(sa.om||0);
        case 'om_asc': return (sa.om||0)-(sb.om||0);
        case 'clay_desc': return (sb.clay||0)-(sa.clay||0);
        case 'n_desc': return (sb.N_lbs_ac||0)-(sa.N_lbs_ac||0);
        default: return a.localeCompare(b);
      }
    });
    if (!filtered.length) continue;
    const collapsed = collapsedGroups.has(groupName);
    container.innerHTML += `
      <div class="fade-in" id="group-${groupName.replace(/\s+/g,'-')}">
        <div class="group-header" onclick="toggleGroup('${groupName}')" style="cursor:pointer">
          <div class="group-name">${groupName}</div><div class="group-count">${filtered.length}</div><div class="group-line"></div>
        </div>
        <div class="fields-grid" style="${collapsed?'display:none':''}">${filtered.map(f=>renderFieldCard(f)).join('')}</div>
      </div>`;
  }
}

function renderFieldCard(fieldName) {
  const s = getFieldSummary(fieldName), isPlanted = !!PLANTING_DB[fieldName];
  if (!s) return '';
  const isSelected = selectedFields.has(fieldName);
  const tColor = TEXTURE_COLORS[s.texture] || '#8a9ab8', tBg = TEXTURE_BG[s.texture] || 'var(--bg2)';

  return `
  <div class="field-card${isSelected?' selected':''}" onclick="openDetail('${fieldName.replace(/'/g,"\\'").replace(/"/g, "&quot;")}')">
    <div class="card-top">
      <div><div class="field-name">${fieldName}</div><div style="font-size:.65rem;color:var(--txt3);font-weight:600">${s.n_samples} pts</div></div>
      <div class="planted-badge ${isPlanted?'planted':'not-planted'}">${isPlanted ? '🌱 '+PLANTING_DB[fieldName] : 'Not Planted'}</div>
    </div>
    <div class="card-metrics">
      <div class="metric full" style="background:${tBg}; border-color:${tColor}30">
        <div class="metric-lbl">Soil Texture</div>
        <div style="display:flex;justify-content:space-between"><span class="texture-pill" style="color:${tColor}">${s.texture}</span></div>
      </div>
      <div class="metric"><div class="metric-lbl">Organic Matter</div><div class="metric-val">${s.om!==null?s.om.toFixed(1)+'%':'—'}</div></div>
      <div class="metric">
        <div class="metric-lbl">Nitrogen (N)</div>
        <div class="metric-val green">${s.N_lbs_ac!==null?Math.round(s.N_lbs_ac)+' lbs/ac':'—'}</div>
        <div style="font-size:.65rem;color:var(--txt3);margin-top:2px">${s.N!==null?Math.round(s.N)+' ppm':'—'}</div>
      </div>
      <div class="metric"><div class="metric-lbl">Hydrological Class</div><div class="metric-val blue">${s.twi_desc}</div></div>
      <div class="metric"><div class="metric-lbl">Topography</div><div class="metric-val">${s.topo_desc}</div></div>
    </div>
    <div style="position:absolute;top:12px;right:12px;width:18px;height:18px;border-radius:50%;border:2px solid ${isSelected?'var(--green)':'var(--border2)'};background:${isSelected?'var(--green)':'transparent'};display:flex;align-items:center;justify-content:center;font-size:.7rem;color:white" onclick="event.stopPropagation();toggleSelect('${fieldName.replace(/'/g,"\\'").replace(/"/g, "&quot;")}')">${isSelected?'✓':''}</div>
  </div>`;
}

function openDetail(fieldName) {
  const s = getFieldSummary(fieldName); if (!s) return;
  let group = '—'; for (const [g,fs] of Object.entries(FIELD_GROUPS)) if (fs.includes(fieldName)) group=g;
  
  document.getElementById('dp-title').textContent = fieldName;
  document.getElementById('dp-sub').innerHTML = `<span>${group}</span>`;
  document.getElementById('dp-body').innerHTML = buildDetailBody(s);
  
  document.getElementById('detail-overlay').classList.add('open'); 
  document.getElementById('detail-panel').classList.add('open');
}

function switchTab(tabId) {
    document.getElementById('img-dem').classList.remove('active');
    document.getElementById('img-twi').classList.remove('active');
    document.getElementById('tab-dem').classList.remove('active');
    document.getElementById('tab-twi').classList.remove('active');
    
    document.getElementById('img-' + tabId).classList.add('active');
    document.getElementById('tab-' + tabId).classList.add('active');
}

function buildDetailBody(s) {
  let visualizerHtml = `<div style="padding: 20px; text-align: center; color: var(--txt3); font-size: 0.85rem; border: 1px dashed var(--border2); border-radius: 8px;">Coordinate data missing or GIS processing failed. Topo rendering unavailable.</div>`;
  
  if (s.dem_b64 && s.twi_b64) {
      visualizerHtml = `
      <div class="map-tabs">
        <button id="tab-dem" class="map-tab active" onclick="switchTab('dem')">Elevation (DEM)</button>
        <button id="tab-twi" class="map-tab" onclick="switchTab('twi')">Moisture (TWI)</button>
      </div>
      <div class="map-image-container">
        <img id="img-dem" class="map-image active" src="data:image/png;base64,${s.dem_b64}" alt="Digital Elevation Model" />
        <img id="img-twi" class="map-image" src="data:image/png;base64,${s.twi_b64}" alt="Topographic Wetness Index" />
      </div>`;
  }

  return `
    <div class="d-section">
      <div class="d-section-title">Core Soil Analytics</div>
      <div class="d-grid">
        <div class="d-stat"><div class="lbl">Texture</div><div class="val">${s.texture}</div></div>
        <div class="d-stat"><div class="lbl">OM %</div><div class="val">${s.om?.toFixed(1)||'—'}</div></div>
        <div class="d-stat"><div class="lbl">pH</div><div class="val">${s.pH?.toFixed(1)||'—'}</div></div>
        <div class="d-stat">
          <div class="lbl">Nitrogen</div>
          <div class="val">${s.N_lbs_ac?.toFixed(0)||'—'} lbs/ac</div>
          <div class="sub">${s.N?.toFixed(1)||'—'} ppm</div>
        </div>
      </div>
    </div>
    
    <div class="d-section">
      <div class="d-section-title">Whitebox LiDAR Profile</div>
      <div class="d-grid" style="margin-bottom: 12px;">
        <div class="d-stat"><div class="lbl">Drainage Profile</div><div class="val blue">${s.twi_desc}</div></div>
        <div class="d-stat"><div class="lbl">Topography</div><div class="val green">${s.topo_desc}</div><div class="sub">Generated from LIO DEM</div></div>
      </div>
      ${visualizerHtml}
    </div>
  `;
}

function closeDetail() { document.getElementById('detail-overlay').classList.remove('open'); document.getElementById('detail-panel').classList.remove('open'); }
function toggleSelect(f) { selectedFields.has(f) ? selectedFields.delete(f) : selectedFields.add(f); updateStats(); renderGroups(); }
function filterFields() { renderGroups(); }
function sortFields() { renderGroups(); }
function toggleGroup(g) { collapsedGroups.has(g) ? collapsedGroups.delete(g) : collapsedGroups.add(g); renderGroups(); }
function toggleAllGroups() {
  const allGroups = Object.keys(FIELD_GROUPS);
  if (collapsedGroups.size === allGroups.length) collapsedGroups.clear();
  else allGroups.forEach(g => collapsedGroups.add(g));
  renderGroups();
}

function exportSelection() {
  const fields = selectedFields.size > 0 ? [...selectedFields] : Object.keys(fieldData);
  const rows = [['Field','Texture','OM %','N ppm','N lbs/ac','Clay %','Sand %','Silt %','pH','Drainage','Topography']];
  fields.forEach(f => {
    const s = getFieldSummary(f); if (!s) return;
    rows.push([f, s.texture, s.om?.toFixed(2)||'', s.N?.toFixed(0)||'', s.N_lbs_ac?.toFixed(0)||'', s.clay?.toFixed(1)||'',
               s.sand?.toFixed(1)||'', s.silt?.toFixed(1)||'', s.pH?.toFixed(1)||'', s.twi_desc, s.topo_desc]);
  });
  const csv = rows.map(r => r.join(',')).join('\n');
  const a = document.createElement('a');
  a.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv);
  a.download = 'upside_field_export.csv'; a.click();
}
// ── INIT ──────────────────────────────────────────────────
document.getElementById('date-badge').textContent = new Date().toLocaleDateString('en-CA',{month:'short',day:'numeric',year:'numeric'});
updateStats(); renderGroups();
</script>
</body>
</html>"""

final_html = HTML_TEMPLATE.replace('__JSON_DATA_HERE__', json_data)
with open(os.path.join(CSV_FOLDER, "Upside_Field_Report.html"), "w", encoding="utf-8") as f: f.write(final_html)

# =========================================================
# 8. PRINT DIAGNOSTIC TRACKER
# =========================================================

print("\n=================================================")
print("🚜 PROCESSING SUMMARY")
print("=================================================")
print(f"✅ Successfully Processed: {len(TRACKER['success'])} fields")

if TRACKER['duplicates_skipped']:
    print(f"\n⚠️ Duplicates Skipped: {len(TRACKER['duplicates_skipped'])} files")
    print("   -> " + ", ".join(TRACKER['duplicates_skipped']))

if TRACKER['missing_kml']:
    print(f"\n⚠️ Missing KML Boundary: {len(TRACKER['missing_kml'])} fields")
    print("   (Used rough square bounding boxes instead)")
    print("   -> " + ", ".join(TRACKER['missing_kml']))

if TRACKER['gis_failed']:
    print(f"\n❌ GIS / LIO API Failed: {len(TRACKER['gis_failed'])} fields")
    print("   -> " + ", ".join(TRACKER['gis_failed']))

if TRACKER['csv_failed']:
    print(f"\n❌ CSV Errors: {len(TRACKER['csv_failed'])} fields")
    print("   -> " + ", ".join(TRACKER['csv_failed']))

print("\n✅ Report generated successfully: Upside_Field_Report.html")
print("=================================================")