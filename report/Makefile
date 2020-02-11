PDF=pdflatex
BIB=bibtex
LATEXOBJS= *.aux *.log *.out *.pdf *.bbl *.blg

RDIR=report
RHEADER=report
RSRCS=$(RDIR)/$(RHEADER).tex
OUTNAME=multimode-report

.PHONY: clean report

report:
	$(PDF) $(RSRCS)
	$(BIB) $(RHEADER)
	$(PDF) $(RSRCS)
	$(PDF) -jobname $(OUTNAME) $(RSRCS)

clean:
	rm -rf $(LATEXOBJS)
