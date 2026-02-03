import pandas as pd
import json
import os
from tqdm import tqdm

# ==========================================
# ⚙️ 設定區
# ==========================================
# 請修改這裡：指向你的 Cityscapes 'vehicle' 資料夾根目錄
# 結構通常是: .../vehicle/train/city_name/xxx_vehicle.json
VEHICLE_ROOT = "/home/rvl1421/Datasets/Cityscapes/sensor_info/vehicle" 

# 輸入與輸出 CSV
INPUT_CSV = "test_final_split.csv"
OUTPUT_CSV = "test_with_gps.csv"
# ==========================================

def get_vehicle_json_path(image_path, vehicle_root):
    """
    解析 image_path 並推導出對應的 vehicle json 路徑
    """
    # 1. 解析路徑結構
    # image_path example: .../leftImg8bit_foggy/train/aachen/aachen_000000_000019_leftImg8bit_foggy_beta_0.005.png
    parts = image_path.split(os.sep)
    
    # 找出 'train', 'val', 'test' 的位置 (split)
    split = None
    city = None
    filename = parts[-1]
    
    # 倒著找比較快，通常結構是 .../split/city/filename
    if len(parts) >= 3:
        city = parts[-2]
        split = parts[-3]
    
    if split not in ['train', 'val', 'test']:
        # 若路徑結構不如預期，嘗試更魯棒的搜尋
        for s in ['train', 'val', 'test']:
            if s in parts:
                split = s
                # 假設 city 在 split 的下一層
                try:
                    idx = parts.index(s)
                    city = parts[idx+1]
                except:
                    pass
                break
    
    if not split or not city:
        print(f"⚠️ 無法解析 split 或 city: {image_path}")
        return None

    # 2. 解析檔案名稱以取得 Base ID (city_seq_frame)
    # Filename: aachen_000000_000019_leftImg8bit_foggy_beta_0.005.png
    # Target: aachen_000000_000019
    name_parts = filename.split('_')
    base_id = "_".join(name_parts[:3]) # 取前三段 (city, seq, frame)
    
    # 3. 組合 Vehicle JSON 路徑
    # 格式: vehicle_root/split/city/base_id_vehicle.json
    json_filename = f"{base_id}_vehicle.json"
    json_path = os.path.join(vehicle_root, split, city, json_filename)
    
    # 有些 dataset 結構可能是 vehicle_trainvaltest/vehicle/train... 需視情況調整
    # 這裡假設 VEHICLE_ROOT 下直接接著 train/val/test
    if not os.path.exists(json_path):
        # 嘗試另一種常見結構: vehicle_root/vehicle/split...
        alt_path = os.path.join(vehicle_root, 'vehicle', split, city, json_filename)
        if os.path.exists(alt_path):
            return alt_path
            
    return json_path

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ 找不到輸入檔案: {INPUT_CSV}")
        return

    print(f"📖 讀取 CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    # 初始化新欄位
    lats = []
    lons = []
    missing_count = 0

    print("🚀 開始匹配 GPS 資料...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row['image_path']
        json_path = get_vehicle_json_path(img_path, VEHICLE_ROOT)
        
        lat, lon = 0.0, 0.0 # 預設值
        
        if json_path and os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    # 讀取 GPS (Cityscapes 格式)
                    lat = data.get('gpsLatitude', 0.0)
                    lon = data.get('gpsLongitude', 0.0)
            except Exception as e:
                print(f"⚠️ 讀取 JSON 錯誤: {json_path}, Error: {e}")
                missing_count += 1
        else:
            # print(f"⚠️ 找不到對應的 JSON: {json_path}") # Debug 用，若太多遺失可打開
            missing_count += 1
            
        lats.append(lat)
        lons.append(lon)

    # 將結果寫入 DataFrame
    df['lat'] = lats
    df['lon'] = lons
    
    print(f"📊 處理完成!")
    print(f"   總筆數: {len(df)}")
    print(f"   遺失 GPS 筆數: {missing_count} (使用預設值 0.0)")
    
    # 儲存
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"💾 已儲存包含 GPS 的 CSV 至: {OUTPUT_CSV}")

    # 檢查前幾筆
    print("\n🔍 預覽前 5 筆資料:")
    print(df[['image_path', 'lat', 'lon']].head())

if __name__ == "__main__":
    main()