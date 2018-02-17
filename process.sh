#!/bin/bash

# To run, this assumes that you are in the directory with the images
# unpacked into train/ and test/.  To run, it works best to use GNU
# parallel as
#
#   ls train/*.jpeg test/*.jpeg | parallel ./process.sh
#
# Otherwise, it also works to do a bash for loop, but this is slower.
#
#   for f in `ls train/*.jpeg test/*.jpeg`; do ./process.sh $f; done
#

# Size of the output image
size=512x512

in_file=$1
out_file=$1

com="convert $in_file \
-set colorspace HSL -channel lightness -equalize \
-set colorspace RGB -channel G -separate -depth 8 \
-convolve Gaussian \
-background black \
-fuzz 10% -trim +repage -resize $size \
-gravity center -background black -extent $size \
-resize $size \
processed/$out_file"

echo $com

$com
