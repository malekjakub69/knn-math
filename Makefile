# Makefile pro kompilaci TeXového dokumentu

# Název dokumentu (bez přípony .tex)
DOCUMENT = dokumentace

# Příkaz pro kompilaci
LATEX = pdflatex
LATEXOPT = -interaction=nonstopmode -file-line-error

# Výchozí cíl - kompilace dokumentu
all: $(DOCUMENT).pdf

# Pravidlo pro kompilaci dokumentu
$(DOCUMENT).pdf: $(DOCUMENT).tex
	$(LATEX) $(LATEXOPT) $(DOCUMENT).tex
	$(LATEX) $(LATEXOPT) $(DOCUMENT).tex  # Druhý průchod pro vygenerování obsahu

# Pravidlo pro odstranění pomocných souborů
clean:
	rm -f $(DOCUMENT).aux $(DOCUMENT).log $(DOCUMENT).toc $(DOCUMENT).out \
		$(DOCUMENT).lof $(DOCUMENT).lot $(DOCUMENT).bbl $(DOCUMENT).blg \
		$(DOCUMENT).nav $(DOCUMENT).snm $(DOCUMENT).vrb $(DOCUMENT).synctex.gz

# Pravidlo pro odstranění všech generovaných souborů včetně PDF
distclean: clean
	rm -f $(DOCUMENT).pdf

# Pravidlo pro zobrazení PDF (funguje na většině Linuxů s příkazem 'xdg-open')
view: $(DOCUMENT).pdf
	xdg-open $(DOCUMENT).pdf 2>/dev/null || open $(DOCUMENT).pdf 2>/dev/null || echo "Nelze otevřít PDF"

# Deklarace, že tyto cíle nejsou soubory
.PHONY: all clean distclean view 