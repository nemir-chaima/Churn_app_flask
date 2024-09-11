import unittest
import pickle
import pandas as pd
import numpy as np


class TestRandomForestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = cls.load_model()

    @staticmethod
    def load_model():
        with open('random_forest_t.pkl', 'rb') as f:
            model = pickle.load(f)
        return model

    @staticmethod
    def generate_input():
        age = np.random.randint(18, 100)
        years = np.random.randint(0, 30)
        num_sites = np.random.randint(0, 50)
        input_data = pd.DataFrame([[age, years, num_sites]])
        return input_data

    def test_model_loaded(self):
        # verifier que le model est bien chargé et qu'il nsagit dun model randomforest
        self.assertIsNotNone(self.model, "Le modèle n'a pas été chargé correctement.")
        self.assertTrue(hasattr(self.model, 'predict'), "Le modèle chargé n'est pas valide.") 

    def test_prediction(self):
        try:
            input_data = self.generate_input()
            # Vérification que les données ne sont pas vides
            self.assertFalse(input_data.empty, "Les données d'entrée générées sont vides.")
            # Faire une prédiction pour verifier si ya des erreurs
            prediction = self.model.predict(input_data)
            # Vérification que la prédiction retourne un tableau numpy
            self.assertIsInstance(prediction, np.ndarray, "La prédiction ne retourne pas un tableau numpy.")

            print(f"Prédiction réussie : {prediction}")
        except Exception as e:
            self.fail(f"Erreur lors de la prédiction : {e}")

if __name__ == "__main__":
    unittest.main()
