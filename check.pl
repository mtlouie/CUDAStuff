#!/usr/bin/perl
while (<>) {
    chomp;
    @parsedline=split / /;
    printf "%s\n", $_ if (abs($parsedline[3]-$parsedline[4])>0);
}
