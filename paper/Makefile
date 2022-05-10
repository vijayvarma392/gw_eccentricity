SHELL := /bin/bash

# Makefile for papers
PAPER := paper

MANUAL_FIGURES :=
AUTO_FIGURES   :=

.DEFAULT_GOAL := $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex $(AUTO_FIGURES) $(MANUAL_FIGURES) References.bib
	pdflatex -file-line-error $<
	bibtex $(PAPER)
	pdflatex -file-line-error $<
	pdflatex -file-line-error $<

# Automatically make pdf files from .gpi (gnuplot) scripts
%.pdf: %.gpi
	gnuplot $<

# Automatically make pdf files from .fig files
%.pdf: %.fig
	fig2mpdf $<

# Automatically make pdf files from .svg files
%.pdf: %.svg
	inkscape -D -z --file=$< --export-pdf=$@

# Automatically make pdf files from .agr files
%.pdf:%.agr
	xmgrace -hardcopy -hdevice EPS -printfile epsfig.eps $<
	epstopdf --outfile=$@ epsfig.eps
	rm -f epsfig.eps

# Automatically make pdf files from .py files
%.pdf:%.py
	python $<

clean:
	@rm -f *.aux
	@rm -f *.out
	@rm -f *.log
	@rm -f *.bbl
	@rm -f *.blg
	@rm -f $(PAPER)Notes.bib
	@rm -f $(AUTO_FIGURES)
	@rm -f $(PAPER).pdf
