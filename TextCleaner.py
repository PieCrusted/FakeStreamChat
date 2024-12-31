import re

class TextCleaner:
    def __init__(self):
        # Define filler words to remove
        # Intentionally doing "oh," to not mess up like "oh no" usage
        self.filler_words = ["um", "uh", "uhm", "uhh", "err", "ah", "huh", "oh wait,", "Oh wait,", "oh,"]

    def clean_text(self, text):
        """
        Clean the input text by:
        - Removing filler words.
        - Removing tokens consisting only of punctuation.
        - Normalizing spaces.
        """
        # Remove filler words
        pattern = r'\b(?:' + '|'.join(re.escape(word) for word in self.filler_words) + r')\b'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Remove tokens consisting only of punctuation
        # This regex ensures that standalone punctuation tokens are removed
        text = re.sub(r'(?<!\w)[^\w\s]+(?!\w)', '', text)

        # Normalize spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

# Example usage
if __name__ == "__main__":
    cleaner = TextCleaner()
    sample_text = "Um, but, uh, grape, grape is barrel Russell's lamp !!!! .... ,,,, grape!!,,"
    print("Original:", sample_text)
    print("Cleaned:", cleaner.clean_text(sample_text))
