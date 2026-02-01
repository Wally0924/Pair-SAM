import pandas as pd
import os

# 這是一個用於根據城市切分資料集的腳本
def split_city_based(input_csv, train_output, test_output, num_test_cities=3):
    print(f"📖 讀取資料: {input_csv}...")
    df = pd.read_csv(input_csv)

    # 1. 解析城市名稱 (取檔名第一個底線前的字串)
    df['city'] = df['image_path'].apply(lambda x: os.path.basename(x).split('_')[0])
    all_cities = sorted(df['city'].unique())

    # 2. 切分城市 (最後 N 個城市做測試)
    test_cities = all_cities[-num_test_cities:]
    train_cities = all_cities[:-num_test_cities]

    # 3. 建立 DataFrame
    train_df = df[df['city'].isin(train_cities)].copy()
    test_df = df[df['city'].isin(test_cities)].copy()

    # 4. 輸出統計
    print("\n📊 資料分割報告:")
    print(f"   - 訓練集城市 ({len(train_cities)}): {train_cities}")
    print(f"   - 測試集城市 ({len(test_cities)}): {test_cities}")
    print(f"   - 訓練集張數: {len(train_df)}")
    print(f"   - 測試集張數: {len(test_df)}")

    # 5. 儲存檔案
    train_df.drop(columns=['city']).to_csv(train_output, index=False)
    test_df.drop(columns=['city']).to_csv(test_output, index=False)
    print(f"✅ 已儲存至: {train_output}, {test_output}")

# 執行
if __name__ == "__main__":
    split_city_based('/home/rvl1421/SAM_research/Datasets/train_all_cached.csv', '/home/rvl1421/SAM_research/Datasets/train_final_split.csv', '/home/rvl1421/SAM_research/Datasets/test_final_split.csv')