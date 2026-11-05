from sklearn.feature_extraction.text import TfidfVectorizer

class LogPreprocessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, logs):
        """Trainiert den Vectorizer und wandelt Text in Zahlen um."""
        return self.vectorizer.fit_transform(logs)

    def transform(self, new_logs):
        """Wandelt neue, unbekannte Logs basierend auf dem gelernten Vokabular um."""
        return self.vectorizer.transform(new_logs)
