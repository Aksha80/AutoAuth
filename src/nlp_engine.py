import spacy
import re

nlp = spacy.load("en_core_web_sm")

class NLPExtractor:

    def extract_features(self, note):

        note_lower = note.lower()

        extracted = {
            "duration_months": 0,
            "severity_score": 1,   # Default Mild
            "previous_therapy": False
        }

        # 1️⃣ Duration detection (digit + month)
        duration_match = re.search(r'(\d+)\s*month', note_lower)
        if duration_match:
            extracted["duration_months"] = int(duration_match.group(1))

        # 2️⃣ Severity detection (simple keyword search)
        if "severe" in note_lower:
            extracted["severity_score"] = 3
        elif "moderate" in note_lower:
            extracted["severity_score"] = 2
        elif "mild" in note_lower:
            extracted["severity_score"] = 1

        # 3️⃣ Therapy detection
        if "therapy" in note_lower or "treatment" in note_lower:
            extracted["previous_therapy"] = True

        return extracted