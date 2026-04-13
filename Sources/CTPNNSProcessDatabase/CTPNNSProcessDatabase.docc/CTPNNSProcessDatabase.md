# ``CTPNNSProcessDatabase``

Process a `.txtpb` PNNS database into an encrypted `.binpb` bundle.

## Overview

`CTPNNSProcessDatabase` is the ciphertext-database analogue of `PNNSProcessDatabase`. Given a
plaintext database in protocol-buffer text format, it:

1. Normalizes and scale-rounds the vectors into a `QuantizedDatabase`.
2. Packs the database rows diagonally (BSGS) into `PlaintextMatrix` values.
3. Generates a fresh BFV secret key and the matching evaluation key (containing the BSGS Galois
   elements and a relinearization key required by the ciphertext × ciphertext kernel).
4. Encrypts each plaintext matrix under that secret key and writes one
   `SerializedCiphertextMatrix` per plaintext modulus.
5. Emits a ready-to-ship directory containing the encrypted database, server config, entry
   metadata, the evaluation key (send with the encrypted DB) and the secret key (keep private).

The resulting directory can be loaded by an offshore server that never sees the secret key; the
server only needs the encrypted database, the evaluation key, and a query ciphertext from the
client to compute encrypted similarity scores.
