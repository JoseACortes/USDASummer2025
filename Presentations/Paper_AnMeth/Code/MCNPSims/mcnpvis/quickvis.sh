#!/bin/bash

# Check if C.txt exists
if [ ! -f "C.txt" ]; then
    echo "Error: C.txt not found in the current directory."
    exit 1
fi

rm comout outp plotm.ps
# Run MCNP plotter on C.txt
mcnp6 ip notek inp=C.txt com=plotcom plotm=plotm


# plotm.ps to png using ghostscript
gs -dNOPAUSE -dBATCH -sDEVICE=pngalpha -r300 -sOutputFile=plotm.png plotm.ps
# rotate the image using gs
gs -dNOPAUSE -dBATCH -sDEVICE=pngalpha -r300 -sOutputFile=plotm.png -c "<</Orientation 3>> setpagedevice" -f plotm.ps