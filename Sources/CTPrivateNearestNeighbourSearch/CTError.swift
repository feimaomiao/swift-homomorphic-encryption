// Copyright 2026 Apple Inc. and the Swift Homomorphic Encryption project authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Copyright 2026 Apple Inc. and the Swift Homomorphic Encryption project authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

public import PrivateNearestNeighborSearch

public enum CTPnnsError: Error, Equatable, Sendable {
    case dimensionMismatch(dbColumns: Int, queryColumns: Int)
    case incorrectSimdRowsCount(got: Int, expected: Int)
    case queryMustBeSingleCiphertext(got: Int)
    case queryMustBeSingleRow(got: Int)
    case wrongContext
    case wrongDatabasePacking(got: MatrixPacking)
    case wrongQueryPacking(got: MatrixPacking)
}
