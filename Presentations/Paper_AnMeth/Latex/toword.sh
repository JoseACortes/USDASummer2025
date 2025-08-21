#!/bin/bash
# Convert main.pdf to main.docx using LibreOffice

input="main.pdf"
output="main.docx"

if [ ! -f "$input" ]; then
    echo "Error: $input not found."
    exit 1
fi

# libreoffice --headless --convert-to docx "$input" --outdir "$(dirname "$input")"

libreoffice --invisible --infilter="writer_pdf_import" --convert-to docx:"Office Open XML Text" "$input"

echo "Conversion complete: $output"