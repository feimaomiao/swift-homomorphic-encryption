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

public import _HomomorphicEncryptionExtras
public import Algorithms
public import AsyncAlgorithms
public import HomomorphicEncryption
public import ModularArithmetic
public import PrivateNearestNeighborSearch

/// Utilities for ciphertext × ciphertext matrix multiplication.
public enum CTMatrixMultiplication {
    /// Evaluation-key configuration for the ciphertext × ciphertext kernel.
    ///
    /// Reuses the plaintext-kernel galois elements and adds a relinearization key, because the
    /// inner product of two ciphertexts returns a degree-3 ciphertext that must be relinearized
    /// before the giant-step rotations.
    @inlinable
    public static func evaluationKeyConfig<Scheme: HeScheme>(
        plaintextMatrixDimensions: MatrixDimensions,
        maxQueryCount: Int,
        encryptionParameters: EncryptionParameters<Scheme.Scalar>,
        scheme: Scheme.Type) throws -> EvaluationKeyConfig
    {
        let plaintextConfig = try MatrixMultiplication.evaluationKeyConfig(
            plaintextMatrixDimensions: plaintextMatrixDimensions,
            maxQueryCount: maxQueryCount,
            encryptionParameters: encryptionParameters,
            scheme: scheme)
        return EvaluationKeyConfig(
            galoisElements: plaintextConfig.galoisElements,
            hasRelinearizationKey: true)
    }
}

extension CiphertextMatrix where Format == Scheme.CanonicalCiphertextFormat {
    /// Ciphertext × ciphertext matrix-vector product using the BSGS algorithm.
    ///
    /// The receiver must be diagonal/BSGS-packed encrypted database rows. `ciphertextVector`
    /// must be a single-row dense-row packed encrypted query. Both must be encrypted under the
    /// same secret key.
    @inlinable
    public func mulTranspose(
        vector ciphertextVector: CiphertextMatrix<Scheme, Scheme.CanonicalCiphertextFormat>,
        using evaluationKey: EvaluationKey<Scheme>) async throws -> [Scheme.CanonicalCiphertext]
    {
        // Preconditions — mirror PlaintextMatrix.mulTranspose(vector:using:) 1:1.
        guard case let .diagonal(babyStepGiantStep: bsgs) = packing else {
            throw CTPnnsError.wrongDatabasePacking(got: packing)
        }
        guard ciphertextVector.packing == .denseRow else {
            throw CTPnnsError.wrongQueryPacking(got: ciphertextVector.packing)
        }
        guard ciphertextVector.context == context else {
            throw CTPnnsError.wrongContext
        }
        guard ciphertextVector.dimensions.columnCount == dimensions.columnCount else {
            throw CTPnnsError.dimensionMismatch(
                dbColumns: dimensions.columnCount,
                queryColumns: ciphertextVector.dimensions.columnCount)
        }
        guard ciphertextVector.dimensions.rowCount == 1 else {
            throw CTPnnsError.queryMustBeSingleRow(got: ciphertextVector.dimensions.rowCount)
        }
        guard ciphertextVector.ciphertexts.count == 1 else {
            throw CTPnnsError.queryMustBeSingleCiphertext(
                got: ciphertextVector.ciphertexts.count)
        }

        // 1) Rotate the encrypted query to produce v_j = theta^j(v) for j in 0..<babyStep.
        //    Stay in canonical form — Scheme.innerProduct(_:_:) takes canonical ciphertexts,
        //    so we skip the plaintext kernel's rotatedCiphertexts → Eval conversion (which
        //    existed purely to speed up plaintext × Eval-ciphertext multiplies).
        let rotatedStates: [Scheme.CanonicalCiphertext] = try await {
            var rotated: [Scheme.CanonicalCiphertext] = []
            rotated.reserveCapacity(bsgs.babyStep)
            var state = ciphertextVector.ciphertexts[0]
            for step in 0..<bsgs.babyStep {
                rotated.append(state)
                if step != bsgs.babyStep - 1 {
                    try await state.rotateColumns(by: -1, using: evaluationKey)
                }
            }
            return rotated
        }()

        let resultCiphertextCount = dimensions.rowCount.dividingCeil(
            context.degree, variableTime: true)
        let dbCiphertexts = ciphertexts

        // Per-(giantStep, resultCiphertext) inner product: pick the right diagonals from the
        // encrypted database, inner-product them against the rotated query states, relinearize
        // (so the result is a 2-poly canonical ciphertext we can rotate in step 3).
        let generateInnerProduct: @Sendable (Int, Int) async throws
            -> Scheme.CanonicalCiphertext = { giantStepIndex, resultCiphertextIndex in
                let ctCount = min(
                    rotatedStates.count,
                    bsgs.vectorDimension - bsgs.babyStep * giantStepIndex)
                let dbRowIndices = (0..<ctCount).map { j in
                    resultCiphertextCount * (j + bsgs.babyStep * giantStepIndex)
                        + resultCiphertextIndex
                }
                let dbRows = dbRowIndices.map { dbCiphertexts[$0] }
                let lhs = Array(rotatedStates.prefix(ctCount))
                var inner = try Scheme.innerProduct(lhs, dbRows)
                try await inner.relinearize(using: evaluationKey)
                return inner
            }

        // 2) For each output ciphertext, combine the giant-step inner products via
        //    rotateColumnsAndSum — same BSGS structure as the plaintext kernel.
        return try await .init((0..<resultCiphertextCount).async
            .map { resultCiphertextIndex in
                let innerProductsToAdd: [Scheme.CanonicalCiphertext] = try await .init(
                    (0..<bsgs.giantStep).async.map { giantStepIndex in
                        try await generateInnerProduct(giantStepIndex, resultCiphertextIndex)
                    })
                return try await Scheme.rotateColumnsAndSumAsync(
                    innerProductsToAdd,
                    by: -bsgs.babyStep,
                    using: evaluationKey)
            })
    }

    /// Batched ciphertext × ciphertext matrix-matrix product using BSGS.
    ///
    /// The receiver must be diagonal/BSGS-packed encrypted database rows. `ciphertextMatrix`
    /// holds one or more dense-row-packed encrypted query vectors. All ciphertexts must be
    /// encrypted under the same secret key. The returned matrix is dense-column packed —
    /// column `j` holds the scores for query row `j`.
    @inlinable
    public func mulTranspose(
        matrix ciphertextMatrix: CiphertextMatrix<Scheme, Scheme.CanonicalCiphertextFormat>,
        using evaluationKey: EvaluationKey<Scheme>) async throws
        -> CiphertextMatrix<Scheme, Scheme.CanonicalCiphertextFormat>
    {
        // Preconditions — mirror PlaintextMatrix.mulTranspose(matrix:using:) 1:1.
        guard dimensions.columnCount == ciphertextMatrix.dimensions.columnCount else {
            throw CTPnnsError.dimensionMismatch(
                dbColumns: dimensions.columnCount,
                queryColumns: ciphertextMatrix.dimensions.columnCount)
        }
        guard ciphertextMatrix.context == context else {
            throw CTPnnsError.wrongContext
        }
        guard case .diagonal = packing else {
            throw CTPnnsError.wrongDatabasePacking(got: packing)
        }
        guard ciphertextMatrix.packing == .denseRow else {
            throw CTPnnsError.wrongQueryPacking(got: ciphertextMatrix.packing)
        }
        guard simdDimensions.rowCount == 2 else {
            throw CTPnnsError.incorrectSimdRowsCount(got: simdDimensions.rowCount, expected: 2)
        }

        // One mulTranspose(vector:) call per query row; extractDenseRow pulls row i out of the
        // batched query ciphertext and replicates it across all SIMD slots so the kernel can
        // treat it like a single-row query.
        let innerProductsChunked: [[Scheme.CanonicalCiphertext]] =
            try await .init((0..<ciphertextMatrix.dimensions.rowCount).async
                .map { rowIndex in
                    let ciphertextRow = try await ciphertextMatrix.extractDenseRow(
                        rowIndex: rowIndex, evaluationKey: evaluationKey)
                    return try await self.mulTranspose(
                        vector: ciphertextRow, using: evaluationKey)
                })
        var innerProducts: [Scheme.CanonicalCiphertext] = innerProductsChunked.flatMap(\.self)

        // Output packing: dense-column matrix whose column count = ciphertextMatrix.dimensions.rowCount.
        // If multiple columns fit in one SIMD row, rotate-and-sum them together and optionally
        // swap-rows-and-add the second SIMD row onto the first to finish one packed ciphertext.
        let columnsPerSimdRowCount = simdDimensions.columnCount / dimensions.rowCount
        if columnsPerSimdRowCount > 0 {
            let columnsPerCiphertextCount = simdDimensions.rowCount * columnsPerSimdRowCount
            let packedCiphertexts: [Scheme.CanonicalCiphertext] =
                try await .init(innerProducts
                    .chunks(ofCount: columnsPerCiphertextCount).async
                    .map { columnsForCiphertext in
                        let packedRows: [Scheme.CanonicalCiphertext] = try await .init(
                            columnsForCiphertext.chunks(ofCount: columnsPerSimdRowCount).async
                                .map { columnsForRow in
                                    try await Scheme.rotateColumnsAndSumAsync(
                                        Array(columnsForRow),
                                        by: dimensions.rowCount,
                                        using: evaluationKey)
                                })
                        if columnsForCiphertext.count > columnsPerSimdRowCount {
                            return try await Scheme.swapRowsAndAddAsync(
                                swapping: packedRows[1],
                                addingTo: packedRows[0],
                                using: evaluationKey)
                        }
                        return packedRows[0]
                    })
            innerProducts = packedCiphertexts
        }

        let resultDimensions = try MatrixDimensions(
            rowCount: dimensions.rowCount,
            columnCount: ciphertextMatrix.dimensions.rowCount)
        return try CiphertextMatrix(
            dimensions: resultDimensions,
            packing: .denseColumn,
            ciphertexts: innerProducts)
    }
}
