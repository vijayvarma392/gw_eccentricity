#!/bin/bash

# 26 Jan 2015 - Leo C. Stein <leo.stein@gmail.com>
# 16 Dec 2015 - edited by LCS
# 9 Feb 2017 - LCS: also ignore commas in "<use ..." strings.

TEMP=`mktemp /tmp/temp.XXXX`

MASTER='paper'

echo ${MASTER}.tex > submission-files
# echo ${MASTER}.bbl >> submission-files
cat extra-files >> submission-files

# get list of figures
grep -o '<use [^>,]*' ${MASTER}.log | awk '{print $2 }' >> submission-files

# make subdirs
grep -o -E '^.+/' submission-files   | sort | uniq > subdirs

mkdir -p submission;
for i in `cat subdirs`; do mkdir -p submission/$i; done;

# copy everything over
for i in `cat submission-files`; do cp -af $i submission/$i; done

# strip the tex files
for i in `find submission -name '*.tex'`; do perl stripcomments.pl $i > $TEMP; mv $TEMP $i; done

# wrap it up
# The COPYFILE_DISABLE environment variable is for mac os x's version
# of tar, which otherwise would include stupid hidden attribute files.
COPYFILE_DISABLE=true tar -c -C submission -T submission-files -z -f ${MASTER}.tar.gz
