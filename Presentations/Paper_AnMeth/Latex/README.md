# In Situ Spectral Analysis for Soil Carbon Measurement - LaTeX Project

This directory contains the LaTeX source files for the academic paper "In Situ Spectral Analysis for Soil Carbon Measurement" using the Elsevier elsarticle document class.

## File Structure

```
USDASummer2025/Presentations/Paper_AnMeth/Latex/
├── main.tex              # Main LaTeX document
├── references.bib        # Bibliography file
├── Makefile             # Build automation
├── build.sh             # Build script
├── README.md            # This file
└── Figures/             # Directory for figures
    └── DataGeneration/  # Figures related to data generation
```

## Prerequisites

Make sure you have the following installed:

- LaTeX distribution (TeXLive, MiKTeX, or MacTeX)
- elsarticle document class (usually included in LaTeX distributions)
- Required packages: lineno, hyperref, amsmath, amssymb, graphicx, booktabs, etc.

## Compilation

### Using Build Script (Recommended)

```bash
# Make the script executable (first time only)
chmod +x build.sh

# Full compilation with bibliography
./build.sh

# Quick compilation without bibliography
./build.sh quick

# Clean auxiliary files
./build.sh clean
```

### Using Make

```bash
# Compile the document with bibliography
make

# Quick compile without bibliography
make quick

# Clean auxiliary files
make clean

# Clean all files including PDF
make cleanall

# View the PDF (Linux)
make view

# Force rebuild
make rebuild
```

### Manual Compilation

```bash
# Full compilation with bibliography
pdflatex main
bibtex main
pdflatex main
pdflatex main

# Quick compilation
pdflatex main
```

## Document Structure

The paper follows the structure outlined in the original markdown document:

1. **Introduction** - Background on soil organic carbon and spectral analysis
2. **Data Generation** - MCNP simulations and soil type modeling
3. **Analysis Methods** - Peak fitting, component fitting, SVD, and deep learning
4. **Results** - Comparison of methods and performance analysis
5. **Discussion** - Conclusions and future work
6. **Acknowledgments and References**

## Figures

Place your figures in the `Figures/` directory maintaining the following structure:

- `Figures/DataGeneration/` - Figures related to data generation section
- Add other subdirectories as needed for different sections

Required figures (referenced in the document):

- `MCNPGeometry.png` - MCNP simulation geometry
- `MCNPSpectralReading.png` - Example spectral reading
- `FeldsparSpectralReadingByCarbonLevel.png` - Feldspar spectra by carbon level
- `Sim_vs_Convoluted_FeldsparSpectralReadings_Combined.png` - Simulation vs convolution comparison

## Customization

### Author Information

Update the author information in the frontmatter section of `main.tex`:

```latex
\author[first]{Your Name\corref{cor1}}
\ead{your.email@university.edu}
```

### Journal

Change the target journal in `main.tex`:

```latex
\journal{Journal of Environmental Science and Technology}
```

### Bibliography Style

The document uses `elsarticle-num` style. To change it, modify the line:

```latex
\bibliographystyle{elsarticle-num}
```

## Adding References

Add new references to `references.bib` using standard BibTeX format:

```bibtex
@article{author2025,
  title={Title of the article},
  author={Author, A. and Author, B.},
  journal={Journal Name},
  volume={XX},
  pages={XXX--XXX},
  year={2025}
}
```

Then cite in the document using `\cite{author2025}`.

## Notes

- The document uses line numbers for review (`\linenumbers`)
- Tables use the `booktabs` package for professional formatting
- Figures use the `[H]` placement to force positioning
- The document is set up for review mode by default

## Troubleshooting

If you encounter compilation errors:

1. Check that all required packages are installed
2. Ensure figure files exist in the correct directories
3. Verify BibTeX entries are properly formatted
4. Run `./build.sh clean` and try compiling again

For missing figures, the compilation will succeed but show placeholder boxes in the PDF.

## Related Files

This LaTeX project is based on the outline found in:
`../outline.md`

Make sure to keep the LaTeX version synchronized with any updates to the outline.
