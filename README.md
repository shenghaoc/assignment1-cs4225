# CS4225 Assignment 1 (Hadoop)

This assignment is an extension of the introductory word count example to find the top k common words between two input files. The ranking is based on either the greater or lower count of the common word between the two files based on the student's matriculation number. For assessment, k is fixed at 20.

Both the word count and, for words with the same word count, the word have to be sorted in reverse order.

## Implementation

The counting itself for both files is done in one job, the corresponding file name is obtained via the context object and recorded during the map stage so that they can be distinguished in the reduce stage.

The secondary sorting problem is delegated to the execution framework in the second job by combining the word count and word into a composite key.

The `REDUCE_OUTPUT_RECORDS` inbuilt counter is used to limit the number of output lines to k.
