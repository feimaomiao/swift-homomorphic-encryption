# ``CTPrivateNearestNeighbourSearch``

Ciphertext × ciphertext prototype for Private Nearest Neighbour Search.

## Overview

This module holds an experimental matrix-vector multiplication kernel where both the database
matrix and the query vector are encrypted under the same BFV secret key. It lets an untrusted
server compute encrypted distance scores without ever seeing database or query plaintexts.
