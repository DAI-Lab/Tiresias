#!/usr/bin/env bash
rm -r docs
pdoc --force --html -o docs tiresias
mv docs/tiresias/* docs
rm -r docs/tiresias