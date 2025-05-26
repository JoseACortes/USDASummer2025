#!/bin/bash

file="pres.tex"  # Remove spaces around '='
output="pres.pdf"

if [ -f "$file" ]; then
    echo "File $file exists. Exporting to PDF..."
    pdflatex -interaction=nonstopmode "$file" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "Export successful. PDF created: $output"
        # Optional: Clean up auxiliary files
        rm -f pres.aux pres.log pres.nav pres.out pres.snm pres.toc
    else
        echo "Export failed. Please check the LaTeX file for errors."
    fi
else
    echo "File $file does not exist. Please check the file path."
fi