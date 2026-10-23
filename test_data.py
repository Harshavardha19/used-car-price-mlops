import unittest
import pandas as pd
from pathlib import Path


class TestCarDataset(unittest.TestCase):

    def test_dataset_exists(self):
        self.assertTrue(
            Path("used_car_prices_new_250.csv").exists()
        )

    def test_required_columns(self):
        df = pd.read_csv("used_car_prices_new_250.csv")

        required_columns = [
            "Brand", "Model", "Year",
            "Kilometers_Driven", "Fuel_Type",
            "Transmission", "Owner_Type",
            "Engine_CC", "Seats", "Selling_Price"
        ]

        for column in required_columns:
            self.assertIn(column, df.columns)

    def test_dataset_has_records(self):
        df = pd.read_csv("used_car_prices_new_250.csv")
        self.assertGreater(len(df), 0)

    def test_selling_price_is_valid(self):
        df = pd.read_csv("used_car_prices_new_250.csv")

        self.assertTrue(df["Selling_Price"].notna().all())
        self.assertTrue((df["Selling_Price"] > 0).all())


if __name__ == "__main__":
    unittest.main()
