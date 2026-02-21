import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")

class NLPExtractor:
    def __init__(self):
        self.matcher = Matcher(nlp.vocab)
        # Define patterns for "Duration" (e.g., "6 months", "2 weeks")
        patterns = [
            [{"IS_DIGIT": True}, {"LOWER": {"IN": ["months", "weeks", "years", "days"]}}],
            [{"LOWER": "severity"}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": {"IN": ["mild", "moderate", "severe"]}}]
        ]
        self.matcher.add("DURATION", [patterns[0]])
        self.matcher.add("SEVERITY", [patterns[1]])

    def extract_features(self, note):
        doc = nlp(note.lower())
        matches = self.matcher(doc)
        
        extracted = {
            "duration_months": 0,
            "severity_score": 1, # 1: Mild, 2: Moderate, 3: Severe
            "previous_therapy": False
        }
        
        for match_id, start, end in matches:
            span = doc[start:end]
            if "month" in span.text:
                extracted["duration_months"] = int(span[0].text)
            if "severe" in span.text:
                extracted["severity_score"] = 3
            elif "moderate" in span.text:
                extracted["severity_score"] = 2

        if "therapy" in note.lower() or "treatment" in note.lower():
            extracted["previous_therapy"] = True
            
        return extracted