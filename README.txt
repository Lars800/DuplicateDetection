This project implements a duplicate detection algorithm  based on
Locality Sensitive Hashing (LSH) to perform approximate matching.
It then implements a version Multi Component Similarity Method+ (Hartveld et al., 2018)
It was written for the Computer Sience for Business Analytics course at Erasmus School of Economics.

To use the code one must first prepare the data as discribed in the
accompanying paper (Recursive LSH for scalable duplicate detection).
It is important maintain the order of the arguments in this dataframe.
This should by Model_ID (for training only, empty otherwise), Title,
 any number of split features (in this case brand and size),
 the full feature list for a product (dictionary),
 Shop


The cleaned data can then be passed to to the get_my_candidates method.
To perform furhter cleaning, a version of the MCSM+ method is implemented in
the generate_duplicates  method.

In the example main, the paper test case is setup.
We run the algorithm for several band sizes, each outputting a different number of candidates.
For robustness we repeat this step 5 times using a booststrap resample of our data.