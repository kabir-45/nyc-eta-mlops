import os
import sys
import pandas as pd
from dataclasses import dataclass
import requests
from pathlib import Path

# Add project root to sys.path to import src.utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils import load_params

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


@dataclass
class DataIngestionConfig:
    raw_data_path: str
    data_type: str = 'yellow'


class DataIngestion:
    def __init__(self, year: int, month: int, config: DataIngestionConfig):
        self.year = year
        self.month = month
        self.config = config

    def _get_url(self):
        file_name = f"{self.config.data_type}_tripdata_{self.year}-{self.month:02}.parquet"
        return f'{BASE_URL}/{file_name}'

    def _download_data(self, url: str, save_path: Path):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading data from {url} to {save_path}")

        resp = requests.get(url, stream=True)
        if resp.status_code != 200:
            raise Exception(f"Failed to download. Status code: {resp.status_code}")

        with open(save_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Saved data to {save_path}")

    def _validate_file(self, path: Path):
        df = pd.read_parquet(path)
        required_columns = [
            'VendorID', 'tpep_pickup_datetime',
            'passenger_count', 'trip_distance', 'RatecodeID', 'store_and_fwd_flag',
            'PULocationID', 'DOLocationID', 'payment_type', 'fare_amount', 'extra',
            'mta_tax', 'tolls_amount', 'improvement_surcharge',
            'total_amount','cbd_congestion_fee'
        ]

        # Check if columns exist (using set intersection for cleaner check)
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise Exception(f"Missing columns in {path}: {missing_cols}")

    def initiate_data_ingestion(self):
        try:
            url = self._get_url()

            # DIRECTLY use the path from params.yaml
            save_path = Path(self.config.raw_data_path)

            self._download_data(url, save_path)
            self._validate_file(save_path)

            return str(save_path)

        except Exception as e:
            raise e


if __name__ == "__main__":
    params = load_params()
    year = params['ingestion']['year']
    month = params['ingestion']['month']
    raw_path = params['ingestion']['raw_data_path']
    config = DataIngestionConfig(raw_data_path=raw_path)
    ingestion = DataIngestion(year=year, month=month, config=config)

    file_path = ingestion.initiate_data_ingestion()