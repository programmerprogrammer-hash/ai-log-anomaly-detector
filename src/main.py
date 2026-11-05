import os
from preprocessor import LogPreprocessor
from detector import AnomalyDetector

def main():
    # Beispiel-Logs direkt im Code zum Testen
    training_logs = [
        "INFO: User admin logged in successfully from IP 192.168.1.5",
        "INFO: Database connection established on port 5432",
        "INFO: User john updated profile settings",
        "INFO: User admin logged in successfully from IP 192.168.1.10",
        "INFO: Fetching data from cache for user maria",
        "CRITICAL: SQL INJECTION DETECTED DROP TABLE USERS FROM IP 99.99.99.99"
    ]

    print("1. Bereite Log-Daten vor...")
    preprocessor = LogPreprocessor()
    X = preprocessor.fit_transform(training_logs)

    print("2. Trainiere das KI-Modell (Isolation Forest)...")
    detector = AnomalyDetector(contamination=0.15)
    detector.train(X)

    print("3. Analysiere Daten auf Anomalien...")
    predictions = detector.predict(X)

    print("\n--- ERGEBNISSE ---")
    for log, pred in zip(training_logs, predictions):
        status = "⚠️ ANOMALIE" if pred == -1 else "✅ NORMAL"
        print(f"[{status}] - {log}")

    # Optional: Modell im 'models/' Ordner sichern
    os.makedirs("models", exist_ok=True)
    detector.save_model("models/trained_model.pkl")
    print("\nModell erfolgreich unter 'models/trained_model.pkl' gespeichert.")

if __name__ == "__main__":
    main()
