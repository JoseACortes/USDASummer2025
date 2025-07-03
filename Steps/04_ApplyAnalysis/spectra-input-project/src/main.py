# Contents of /spectra-input-project/spectra-input-project/src/main.py

import sys
from spectra.input_handler import InputHandler

def main():
    print("Welcome to the Spectra Input Project!")
    input_handler = InputHandler()

    while True:
        choice = input("Would you like to input a spectrum (y/n)? ").strip().lower()
        if choice == 'y':
            input_handler.collect_spectrum()
        elif choice == 'n':
            print("Exiting the program.")
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

if __name__ == "__main__":
    main()