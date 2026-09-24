LATEX=pdflatex
MAIN=main
NEW=main_new

.PHONY: all new clean distclean view

all: $(MAIN).pdf

new: $(NEW).pdf

$(MAIN).pdf: $(MAIN).tex $(MAIN).bbl mlsys2026.sty tex/*.tex
	$(LATEX) -interaction=nonstopmode $(MAIN)
	$(LATEX) -interaction=nonstopmode $(MAIN)

$(NEW).pdf: $(NEW).tex $(NEW).bbl mlsys2026.sty tex_new/*.tex
	$(LATEX) -interaction=nonstopmode $(NEW)
	$(LATEX) -interaction=nonstopmode $(NEW)

view: $(MAIN).pdf
	open $(MAIN).pdf

clean:
	rm -f $(MAIN).aux $(MAIN).blg $(MAIN).log \
	      $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot \
	      $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).synctex.gz \
	      $(NEW).aux $(NEW).blg $(NEW).log \
	      $(NEW).out $(NEW).toc $(NEW).lof $(NEW).lot \
	      $(NEW).fls $(NEW).fdb_latexmk $(NEW).synctex.gz \
	      tex/*.aux tex_new/*.aux

distclean: clean
	rm -f $(MAIN).pdf
