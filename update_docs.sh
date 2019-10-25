#!/usr/bin/env bash
pdoc --force --html -o docs tiresias
mv docs/tiresias/* docs
rm -r docs/tiresias
