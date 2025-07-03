class Spectrum:
    def __init__(self, data):
        if len(data) != 1024:
            raise ValueError("Spectrum must be a 1024-dimensional vector.")
        self.data = data

    def normalize(self):
        """Normalize the spectrum data to a range of [0, 1]."""
        min_val = self.data.min()
        max_val = self.data.max()
        self.data = (self.data - min_val) / (max_val - min_val)

    def get_peak(self):
        """Return the index of the maximum value in the spectrum."""
        return self.data.argmax()

    def __repr__(self):
        return f"Spectrum(data={self.data})"