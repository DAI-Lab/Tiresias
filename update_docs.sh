#!/usr/bin/env bash
rm -r docs
pdoc --force --html -o docs tiresias
mv docs/tiresias/* docs
cp -r images docs/images
rm -r docs/tiresias

# remove header from generated file
python -c 'import re; \
html = open("docs/index.html").read(); \
html = re.sub(r"<header>.+</header>", "", html, flags=re.DOTALL); \
open("docs/index.html", "wt").write(html);'
