import joblib
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self, contamination=0.15):
        self.model = IsolationForest(contamination=contamination, random_state=42)

    def train(self, X):
        """Trainiert das Modell mit den vorbereiteten Vektoren."""
        self.model.fit(X)

    def predict(self, X):
        """Sagt voraus, ob es sich um eine Anomalie (-1) oder Normalfall (1) handelt."""
        return self.model.predict(X)

    def save_model(self, file_path):
        """Speichert das fertige Modell auf der Festplatte."""
        joblib.dump(self.model, file_path)

    def load_model(self, file_path):
        """Lädt ein bereits trainiertes Modell."""
        self.model = joblib.load(file_path)
