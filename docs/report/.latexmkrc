# latexmk configuration for intermediate-progress-report.tex
# Run from inside docs/report/:  latexmk        → builds PDF
#                                latexmk -c     → cleans intermediates
#                                latexmk -C     → cleans everything including PDF

$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error -file-line-error %O %S';
$bibtex_use = 2;
$out_dir = '.';
@default_files = ('intermediate-progress-report.tex');

# Extra files to clean with -c
$clean_ext = 'bbl bcf run.xml fls fdb_latexmk synctex.gz';
