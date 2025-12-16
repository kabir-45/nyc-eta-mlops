import pandas as pd
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class BuildFeatures:
    drop_invalid_rows: bool = True
    lower_bounds: Dict[str, float] = field(default_factory=dict)
    upper_bounds: Dict[str, float] = field(default_factory=dict)

    def fit(self, df, y=None):
        temp_df = df.copy()
        temp_df['tpep_pickup_datetime'] = pd.to_datetime(temp_df['tpep_pickup_datetime'])

        if 'tpep_dropoff_datetime' in temp_df.columns:
            temp_df['tpep_dropoff_datetime'] = pd.to_datetime(temp_df['tpep_dropoff_datetime'])
            temp_df['ETA'] = (temp_df['tpep_dropoff_datetime'] - temp_df['tpep_pickup_datetime']).dt.total_seconds() / 60

        columns_to_clip = ['trip_distance', 'ETA']

        for col in columns_to_clip:
            if col in temp_df.columns:
                self.lower_bounds[col] = temp_df[col].quantile(0.01)
                self.upper_bounds[col] = temp_df[col].quantile(0.99)

        if 'ETA' in self.lower_bounds:
            self.lower_bounds['ETA'] = max(self.lower_bounds['ETA'], 1.0)
        if 'trip_distance' in self.lower_bounds:
            self.lower_bounds['trip_distance'] = max(self.lower_bounds['trip_distance'], 0.1)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.drop_invalid_rows:
            df.dropna(inplace=True)

        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])

        if 'tpep_dropoff_datetime' in df.columns:
            df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
            df['ETA'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime'])
            df['ETA'] = df['ETA'].dt.total_seconds() / 60.0

            if 'ETA' in self.upper_bounds:
                mask = (df['ETA'] >= self.lower_bounds['ETA']) & (df['ETA'] <= self.upper_bounds['ETA'])
                df = df[mask]

        if 'trip_distance' in self.upper_bounds:
            mask = (df['trip_distance'] >= self.lower_bounds['trip_distance']) & \
                   (df['trip_distance'] <= self.upper_bounds['trip_distance'])
            df = df[mask]

        df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
        df['day_of_week'] = df['tpep_pickup_datetime'].dt.weekday

        columns_to_drop = [
            'tpep_dropoff_datetime',
            'tpep_pickup_datetime',
            'fare_amount', 'total_amount', 'mta_tax', 'payment_type',
            'improvement_surcharge', 'tip_amount', 'tolls_amount', 'extra',
            'congestion_surcharge', 'Airport_fee', 'cbd_congestion_fee', 'store_and_fwd_flag'
        ]

        for col in columns_to_drop:
            if col in df.columns:
                df.drop(columns=col, inplace=True)

        return df