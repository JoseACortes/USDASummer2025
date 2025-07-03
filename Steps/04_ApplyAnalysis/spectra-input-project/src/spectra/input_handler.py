class InputHandler:
    def __init__(self):
        self.spectra = []

    def add_spectrum(self, spectrum):
        if len(spectrum) == 1024:
            self.spectra.append(spectrum)
        else:
            raise ValueError("Spectrum must be a 1024-dimensional vector.")

    def read_spectra_from_file(self, file_path):
        import numpy as np
        try:
            data = np.loadtxt(file_path)
            for spectrum in data:
                self.add_spectrum(spectrum)
        except Exception as e:
            print(f"Error reading spectra from file: {e}")

    def input_spectra(self, num_spectra):
        for i in range(num_spectra):
            spectrum = list(map(float, input(f"Enter spectrum {i + 1} (1024 values separated by spaces): ").split()))
            self.add_spectrum(spectrum)

    def get_spectra(self):
        return self.spectra