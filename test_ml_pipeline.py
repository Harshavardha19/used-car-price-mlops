import json
import unittest
from pathlib import Path

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_model_exists(self):
        self.assertTrue(Path("used_car_price_model.pkl").exists())

    def test_metrics_exist(self):
        self.assertTrue(Path("metrics.json").exists())

    def test_metrics_are_valid(self):
        with open("metrics.json", encoding="utf-8") as file:
            metrics = json.load(file)

        for key in ["mae", "rmse", "r2"]:
            self.assertIn(key, metrics)

        self.assertGreaterEqual(metrics["mae"], 0)
        self.assertGreaterEqual(metrics["rmse"], 0)

    def test_model_predicts_price(self):
        model = joblib.load("used_car_price_model.pkl")

        sample = pd.DataFrame([{
            "Brand": "Maruti",
            "Model": "Swift",
            "Year": 2020,
            "Kilometers_Driven": 30000,
            "Fuel_Type": "Petrol",
            "Transmission": "Manual",
            "Owner_Type": "First",
            "Engine_CC": 1200,
            "Seats": 5,
        }])

        prediction = model.predict(sample)

        self.assertEqual(len(prediction), 1)
        self.assertGreater(float(prediction[0]), 0)


if __name__ == "__main__":
    unittest.main()
