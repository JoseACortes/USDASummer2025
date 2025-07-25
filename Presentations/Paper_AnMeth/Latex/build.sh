#!/bin/bash

# Build script for the LaTeX paper
# Usage: ./build.sh [clean|quick|full]

set -e

MAIN="main"

case "${1:-full}" in
    "clean")
        echo "Cleaning auxiliary files..."
        rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
        echo "Clean complete."
        ;;
    "quick")
        echo "Quick compilation (no bibliography)..."
        echo "Running: pdflatex $MAIN"
        pdflatex "$MAIN"
        echo "Quick compilation complete."
        ;;
    "full"|*)
        echo "Full compilation with bibliography..."
        echo "Step 1: Running pdflatex $MAIN"
        pdflatex "$MAIN"
        if [ -f "$MAIN.aux" ]; then
            echo "Step 2: Running bibtex $MAIN"
            bibtex "$MAIN"
            echo "Step 3: Running pdflatex $MAIN (second pass)"
            pdflatex "$MAIN"
            echo "Step 4: Running pdflatex $MAIN (final pass)"
            pdflatex "$MAIN"
        fi
        echo "Full compilation complete."
        ;;
esac

if [ -f "$MAIN.pdf" ]; then
    echo "Success! Output: $MAIN.pdf"
else
    echo "Error: PDF not generated"
    exit 1
fi
