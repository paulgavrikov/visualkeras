.PHONY: help clean html dirhtml singlehtml pickle json htmlhelp qthelp devhelp epub latex latexpdf text man changes linkcheck doctest coverage gettext

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  dirhtml    to make HTML files named index.html in directories"
	@echo "  singlehtml to make a single large HTML file"
	@echo "  pickle     to make pickle files"
	@echo "  json       to make JSON files"
	@echo "  htmlhelp   to make HTML files and an HTML help project"
	@echo "  qthelp     to make HTML files and a qthelp project"
	@echo "  devhelp    to make HTML files and a Devhelp project"
	@echo "  epub       to make an epub file"
	@echo "  latex      to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  latexpdf   to make LaTeX files and run them through pdflatex"
	@echo "  text       to make text files"
	@echo "  man        to make manual pages"
	@echo "  texinfo    to make Texinfo files"
	@echo "  info       to make Texinfo files and run them through makeinfo"
	@echo "  gettext    to make PO message catalogs"
	@echo "  changes    to make an overview of all changed/added/deprecated items"
	@echo "  linkcheck  to make external links check their validity"
	@echo "  doctest    to make sure the docstrings in the Python code are correct"
	@echo "  coverage   to run coverage check of the documentation"

clean:
	rm -rf build/*

html:
	sphinx-build -b html -d build/doctrees source build/html

dirhtml:
	sphinx-build -b dirhtml -d build/doctrees source build/dirhtml

singlehtml:
	sphinx-build -b singlehtml -d build/doctrees source build/singlehtml

pickle:
	sphinx-build -b pickle -d build/doctrees source build/pickle

json:
	sphinx-build -b json -d build/doctrees source build/json

htmlhelp:
	sphinx-build -b htmlhelp -d build/doctrees source build/htmlhelp

qthelp:
	sphinx-build -b qthelp -d build/doctrees source build/qthelp

devhelp:
	sphinx-build -b devhelp -d build/doctrees source build/devhelp

epub:
	sphinx-build -b epub -d build/doctrees source build/epub

latex:
	sphinx-build -b latex -d build/doctrees source build/latex

latexpdf:
	sphinx-build -b latex -d build/doctrees source build/latex
	cd build/latex && make all-pdf

text:
	sphinx-build -b text -d build/doctrees source build/text

man:
	sphinx-build -b man -d build/doctrees source build/man

texinfo:
	sphinx-build -b texinfo -d build/doctrees source build/texinfo

info:
	sphinx-build -b texinfo -d build/doctrees source build/texinfo
	cd build/texinfo && make info

gettext:
	sphinx-build -b gettext -d build/doctrees source build/gettext

changes:
	sphinx-build -b changes -d build/doctrees source build/changes

linkcheck:
	sphinx-build -b linkcheck -d build/doctrees source build/linkcheck

doctest:
	sphinx-build -b doctest -d build/doctrees source build/doctest

coverage:
	sphinx-build -b coverage -d build/doctrees source build/coverage

serve:
	cd build/html && python -m http.server 8000
