# Spectra Input Project

This project allows users to input multiple spectra, where each spectrum is represented as a 1024-dimensional vector. The application is designed to handle user input efficiently and manage the spectra data effectively.

## Project Structure

```
spectra-input-project
├── src
│   ├── main.py               # Entry point of the application
│   ├── spectra
│   │   ├── __init__.py       # Package initialization
│   │   └── input_handler.py   # Handles input of multiple spectra
│   └── types
│       └── spectrum.py        # Defines the Spectrum class
├── requirements.txt           # Lists project dependencies
└── README.md                  # Project documentation
```

## Installation

To set up the project, clone the repository and navigate to the project directory. Then, install the required dependencies using:

```
pip install -r requirements.txt
```

## Usage

To run the application, execute the following command:

```
python src/main.py
```

Follow the prompts to input your spectra data. The application will handle multiple spectra and store them for further analysis.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.