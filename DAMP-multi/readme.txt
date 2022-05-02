# DAMP-balanced dataset #

## Description ##
The DAMP-balanced dataset contains 24874 solo singing performances from 5429 singers singing a collection of 14 songs.
The structure of the DAMP-balanced is that the last 4 songs are designed to be the test set, and the first 10 songs
could be partitioned into any 6/4 train/validation split (permutation) that the singers in train and validation set sang
the same 6/4 songs collections according to the 6/4 split (the number of total recordings for train and validation set
are different from split to split, since there are different number of singers that all sang the same 6/4 split for
different split).

For example, a subset from the dataset splitting the 14 songs into 6/4/4 train/validation/test sets having 276/88/224
performances sang by 46/22/56 singers could be extracted from the dataset and be used for machine learning tasks.
Each singer in the train, validation and test set, sang each of the 6/4/4 songs once respectively, thus making the
collections of performances "balanced" respect to the song collections.

## List of songs ##
1. one call away - Charlie Puth(3912)
2. say you won`t let go - James Arthur(3255)
3. all of me - John Legend(2856)
4. closer -The Chainsmokers(2873)
5. seven years - Lukas Graham(2942)
6. despacito - Luis Fonsi(1287)
7. more than words - Extreme(586)
8. lost boy - Ruth B.(1183)
9. love yourself - Justin Bieber(3019)
10. rockabye - Clean Bandit(2737)
11. part of your world - Jodi Benson(56)
12. when I was your man - Bruno Mars(56)
13. chandelier - Sia(56)
14. cups - Anna Kendrick(56)


