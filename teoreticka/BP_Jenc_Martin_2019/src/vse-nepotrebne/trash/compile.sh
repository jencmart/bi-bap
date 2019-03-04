#!/bin/bash

pdflatex ./BP_Jenc_Martin_2019.tex

rm ./BP_Jenc_Martin_2019.aux
rm ./BP_Jenc_Martin_2019.lof
rm ./BP_Jenc_Martin_2019.out
rm ./BP_Jenc_Martin_2019.toc
rm ./BP_Jenc_Martin_2019.log

google-chrome-stable BP_Jenc_Martin_2019.pdf


