import pandas as pd
from io import StringIO
from functools import reduce
import numpy as np
def load_csv_data(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        return f.read()

file_paths = {
    'table73': '/content/sample_data/table73_eng.csv',
    'table71': '/content/sample_data/table71_eng.csv',
    'table54': '/content/sample_data/table54_eng.csv',
    'table51a': '/content/sample_data/table51a_eng.csv',
    'table51b': '/content/sample_data/table51b_eng.csv',
    'table51e': '/content/sample_data/table51e_eng.csv',
    'table44': '/content/sample_data/table44_eng.csv',
    'table45': '/content/sample_data/table45_eng.csv',
    'table72': '/content/sample_data/table72_eng.csv'

}
def preprocess_and_merge(csv_contents_dict,iqr_multiplier=1.5,
    rate_clip=(0, 100)):
    processed_dfs = []
    for file_name, content in csv_contents_dict.items():
        # read csv files
        df = pd.read_csv(
            StringIO(content),
            encoding='utf-8-sig',
            na_values=['', 'NA', '#N/A', 'NaN', 'NULL', ' ', '#'],
            thousands=','
        )

        yrmth_candidates = [c for c in df.columns if 'YR_MTH' in c.strip()]

        if not yrmth_candidates:
            raise ValueError(
                f"File {file_name} lack key time YR_MTH，actual: {df.columns.tolist()}"
            )
        yrmth_col = yrmth_candidates[0]
        # ----------------------------------


        df['DATE'] = pd.to_datetime(df[yrmth_col], format='%Y%m', errors='coerce')


        if df['DATE'].isnull().any():
            bad_dates = df[df['DATE'].isnull()][yrmth_col].unique()
            raise ValueError(
                f"File {file_name} unknown format: {bad_dates}，example 201301"
            )


        if 'table51a' in file_name:
            df = df.pivot(
                index='DATE',
                columns='DRI_LICENSE_HOLDER_TYPE_CODE',
                values=['NO_VALID_LIC', 'NO_NOVER_3YR_EXP_LIC', 'NO_OVER_3YR_EXP_LIC']
            )
            df.columns = [f'{col[0]}_TYPE{col[1]}' for col in df.columns]
            df = df.reset_index()
        else:
            indi_cols = [c for c in df.columns if '_INDI' in c]
            cols_to_drop = indi_cols + [yrmth_col]
            df = df.drop(columns=cols_to_drop, errors='ignore').drop_duplicates()

        processed_dfs.append(df)

    # Based on time to merge
    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on='DATE', how='outer'),
        processed_dfs
    )
    merged_df = merged_df.dropna(axis=1, how='all')
    try:
      # Mapp
      vehicle_col = 'VCT_VEHICLE'        # From table71
      probat_col = 'NO_ISSUE_WARN_8_14PT'  # From table54

      # Check
      required_cols = [vehicle_col, probat_col]
      missing = [c for c in required_cols if c not in merged_df.columns]
      if missing:
          raise KeyError(f"Missing: {missing}，Can use：{merged_df.columns.tolist()}")

      # Outliers using IQR
      mask = pd.Series(True, index=merged_df.index)
      for col in required_cols:
          q1 = merged_df[col].quantile(0.25)
          q3 = merged_df[col].quantile(0.75)
          iqr = q3 - q1
          lower = q1 - 1.5*iqr
          upper = q3 + 1.5*iqr
          mask &= merged_df[col].between(lower, upper)

      merged_df = merged_df[mask]

      # Compute the accident rate
      merged_df['ACCIDENT_RATE'] = (
          merged_df[vehicle_col]
          / merged_df[probat_col].replace(0, np.nan)
      )

      # Clean the none value
      merged_df['ACCIDENT_RATE'] = (
          merged_df['ACCIDENT_RATE']
          .replace([np.inf, -np.inf], np.nan)
          .fillna(0)
          .clip(lower=0, upper=1)  # Normolization
      )

    except Exception as e:
        raise RuntimeError(f"Failed: {str(e)}") from e

    return (
        merged_df
        .sort_values('DATE')
        .dropna(axis=1, how='all')
        .reset_index(drop=True)
    )

csv_contents_dict = {name: load_csv_data(path) for name, path in file_paths.items()}
merged_data = preprocess_and_merge(csv_contents_dict)

print(merged_data[['DATE', 'VCT_VEHICLE', 'NO_ISSUE_WARN_8_14PT', 'ACCIDENT_RATE']].head(3))
merged_data.to_csv(
    'merged_traffic_data.csv',  # 文件名
    index=False,               # 不保存索引列
    encoding='utf-8-sig'       # 支持中文的编码格式
)