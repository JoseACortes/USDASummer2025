# Makefile for LaTeX document compilation
# Usage: make, make clean, make view

MAIN = main
TEXFILES = $(MAIN).tex
BIBFILES = references.bib
PDFLATEX = pdflatex
BIBTEX = bibtex

# Default target
all: $(MAIN).pdf

# Compile PDF
$(MAIN).pdf: $(TEXFILES) $(BIBFILES)
	$(PDFLATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(PDFLATEX) $(MAIN)
	$(PDFLATEX) $(MAIN)

# Quick compile (without bibliography)
quick: $(TEXFILES)
	$(PDFLATEX) $(MAIN)

# Clean auxiliary files
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

# Clean all generated files including PDF
cleanall: clean
	rm -f $(MAIN).pdf

# View PDF (requires a PDF viewer)
view: $(MAIN).pdf
	xdg-open $(MAIN).pdf &

# Force rebuild
rebuild: cleanall all

.PHONY: all quick clean cleanall view rebuild
