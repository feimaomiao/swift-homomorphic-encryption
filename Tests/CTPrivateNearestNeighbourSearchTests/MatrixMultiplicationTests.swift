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

import ApplicationProtobuf
import CTPrivateNearestNeighbourSearch
import Foundation
import HomomorphicEncryption
import HomomorphicEncryptionProtobuf
import PrivateNearestNeighborSearch
import Testing
#if canImport(Darwin)
import Darwin
#endif

struct CTMatrixMultiplicationTests {
    private struct CosineSimilarityParityInputs {
        let context: Bfv<UInt64>.Context
        let parameters: EncryptionParameters<UInt64>
        let scalingFactor: Float
        let rowCount: Int
        let columnCount: Int
        let seed: UInt64
    }

    @Test
    func evaluationKeyConfigHasRelinearizationKey() throws {
        let parameters = try EncryptionParameters<UInt64>(from: .n_4096_logq_27_28_28_logt_16)
        let plaintextDims = try MatrixDimensions(rowCount: 16, columnCount: 16)

        let config = try CTMatrixMultiplication.evaluationKeyConfig(
            plaintextMatrixDimensions: plaintextDims,
            maxQueryCount: 1,
            encryptionParameters: parameters,
            scheme: Bfv<UInt64>.self)

        #expect(config.hasRelinearizationKey)

        let plaintextConfig = try MatrixMultiplication.evaluationKeyConfig(
            plaintextMatrixDimensions: plaintextDims,
            maxQueryCount: 1,
            encryptionParameters: parameters,
            scheme: Bfv<UInt64>.self)
        for galois in plaintextConfig.galoisElements {
            #expect(config.galoisElements.contains(galois))
        }
    }

    @Test
    func mulVectorMatchesPlaintextKernel() async throws {
        try await runMulVectorParity(for: Bfv<UInt32>.self)
        try await runMulVectorParity(for: Bfv<UInt64>.self)
    }

    private func runMulVectorParity<Scheme: HeScheme>(for _: Scheme.Type) async throws {
        let encryptionParameters = try EncryptionParameters<Scheme.Scalar>(
            from: .n_4096_logq_27_28_28_logt_16)
        let context = try Scheme.Context(encryptionParameters: encryptionParameters)
        let secretKey = try context.generateSecretKey()

        let rowCount = 16
        let columnCount = 16
        let plaintextRows: [[Scheme.Scalar]] = (0..<rowCount).map { r in
            (0..<columnCount).map { c in Scheme.Scalar((r + c) % 7) }
        }
        let queryValues: [Scheme.Scalar] = (0..<columnCount).map { Scheme.Scalar($0 % 5) }

        let dims = try MatrixDimensions(rowCount: rowCount, columnCount: columnCount)
        let bsgs = BabyStepGiantStep(vectorDimension: columnCount)

        // --- Plaintext reference path ---
        let dbPlaintextMatrix = try PlaintextMatrix<Scheme, Coeff>(
            context: context,
            dimensions: dims,
            packing: .diagonal(babyStepGiantStep: bsgs),
            values: plaintextRows.flatMap(\.self)).convertToEvalFormat()

        let queryCiphertextDims = try MatrixDimensions(rowCount: 1, columnCount: columnCount)
        let queryCiphertext = try PlaintextMatrix<Scheme, Coeff>(
            context: context,
            dimensions: queryCiphertextDims,
            packing: .denseRow,
            values: queryValues).encrypt(using: secretKey)

        let plaintextKernelConfig = try MatrixMultiplication.evaluationKeyConfig(
            plaintextMatrixDimensions: dims,
            maxQueryCount: 1,
            encryptionParameters: encryptionParameters,
            scheme: Scheme.self)
        let plaintextKernelKey = try context.generateEvaluationKey(
            config: plaintextKernelConfig, using: secretKey)
        let plaintextResultCts = try await dbPlaintextMatrix.mulTranspose(
            vector: queryCiphertext, using: plaintextKernelKey)

        var plaintextDecoded: [Scheme.Scalar] = []
        for ct in plaintextResultCts {
            let pt = try ct.decrypt(using: secretKey)
            plaintextDecoded += try pt.decode(format: .simd) as [Scheme.Scalar]
        }

        // --- Ciphertext kernel under test ---
        let dbCiphertextMatrix = try PlaintextMatrix<Scheme, Coeff>(
            context: context,
            dimensions: dims,
            packing: .diagonal(babyStepGiantStep: bsgs),
            values: plaintextRows.flatMap(\.self)).encrypt(using: secretKey)

        let ctConfig = try CTMatrixMultiplication.evaluationKeyConfig(
            plaintextMatrixDimensions: dims,
            maxQueryCount: 1,
            encryptionParameters: encryptionParameters,
            scheme: Scheme.self)
        let ctKey = try context.generateEvaluationKey(config: ctConfig, using: secretKey)

        let ctResultCts = try await dbCiphertextMatrix.mulTranspose(
            vector: queryCiphertext, using: ctKey)
        var ctDecoded: [Scheme.Scalar] = []
        for ct in ctResultCts {
            let pt = try ct.decrypt(using: secretKey)
            ctDecoded += try pt.decode(format: .simd) as [Scheme.Scalar]
        }

        #expect(ctDecoded == plaintextDecoded, "C×C kernel diverged from plaintext kernel")
    }

    @Test
    func mulVectorTallMatchesPlaintextKernel() async throws {
        let degree = 2048
        let coefficientModuli = try UInt64.generatePrimes(
            significantBitCounts: [29, 29, 29, 29],
            preferringSmall: false,
            nttDegree: degree)
        let plaintextModulus = try UInt64.generatePrimes(
            significantBitCounts: [16],
            preferringSmall: true,
            nttDegree: degree)[0]
        let parameters = try EncryptionParameters<UInt64>(
            polyDegree: degree,
            plaintextModulus: plaintextModulus,
            coefficientModuli: coefficientModuli,
            errorStdDev: .stdDev32,
            securityLevel: .unchecked)
        let context = try Bfv<UInt64>.Context(encryptionParameters: parameters)
        let secretKey = try context.generateSecretKey()

        // rowCount > degree forces resultCiphertextCount > 1.
        let rowCount = 2 * degree
        let columnCount = 16
        let plaintextRows: [[UInt64]] = (0..<rowCount).map { r in
            (0..<columnCount).map { c in UInt64((r &+ c) % 5) }
        }
        let queryValues: [UInt64] = (0..<columnCount).map { UInt64($0 % 3) }

        let dims = try MatrixDimensions(rowCount: rowCount, columnCount: columnCount)
        let bsgs = BabyStepGiantStep(vectorDimension: columnCount)

        let dbPlaintext = try PlaintextMatrix<Bfv<UInt64>, Coeff>(
            context: context, dimensions: dims,
            packing: .diagonal(babyStepGiantStep: bsgs),
            values: plaintextRows.flatMap(\.self)).convertToEvalFormat()
        let dbCiphertext = try PlaintextMatrix<Bfv<UInt64>, Coeff>(
            context: context, dimensions: dims,
            packing: .diagonal(babyStepGiantStep: bsgs),
            values: plaintextRows.flatMap(\.self)).encrypt(using: secretKey)

        let queryDims = try MatrixDimensions(rowCount: 1, columnCount: columnCount)
        let queryCiphertext = try PlaintextMatrix<Bfv<UInt64>, Coeff>(
            context: context, dimensions: queryDims,
            packing: .denseRow, values: queryValues).encrypt(using: secretKey)

        let plaintextKey = try context.generateEvaluationKey(
            config: MatrixMultiplication.evaluationKeyConfig(
                plaintextMatrixDimensions: dims, maxQueryCount: 1,
                encryptionParameters: parameters, scheme: Bfv<UInt64>.self),
            using: secretKey)
        let ctKey = try context.generateEvaluationKey(
            config: CTMatrixMultiplication.evaluationKeyConfig(
                plaintextMatrixDimensions: dims, maxQueryCount: 1,
                encryptionParameters: parameters, scheme: Bfv<UInt64>.self),
            using: secretKey)

        let plaintextOut = try await dbPlaintext.mulTranspose(
            vector: queryCiphertext, using: plaintextKey)
        let ctOut = try await dbCiphertext.mulTranspose(
            vector: queryCiphertext, using: ctKey)

        #expect(plaintextOut.count == ctOut.count)
        for (p, c) in zip(plaintextOut, ctOut) {
            let pDec: [UInt64] = try p.decrypt(using: secretKey).decode(format: .simd)
            let cDec: [UInt64] = try c.decrypt(using: secretKey).decode(format: .simd)
            #expect(pDec == cDec)
        }
    }

    @Test
    func mulVectorHasPositiveNoiseBudget() async throws {
        let parameters = try EncryptionParameters<UInt64>(from: .n_4096_logq_27_28_28_logt_16)
        let context = try Bfv<UInt64>.Context(encryptionParameters: parameters)
        let secretKey = try context.generateSecretKey()

        let dims = try MatrixDimensions(rowCount: 16, columnCount: 16)
        let bsgs = BabyStepGiantStep(vectorDimension: 16)
        let values: [UInt64] = Array(repeating: 1, count: 16 * 16)
        let query: [UInt64] = Array(repeating: 1, count: 16)

        let db = try PlaintextMatrix<Bfv<UInt64>, Coeff>(
            context: context, dimensions: dims,
            packing: .diagonal(babyStepGiantStep: bsgs),
            values: values).encrypt(using: secretKey)
        let queryDims = try MatrixDimensions(rowCount: 1, columnCount: 16)
        let queryCt = try PlaintextMatrix<Bfv<UInt64>, Coeff>(
            context: context, dimensions: queryDims,
            packing: .denseRow, values: query).encrypt(using: secretKey)

        let key = try context.generateEvaluationKey(
            config: CTMatrixMultiplication.evaluationKeyConfig(
                plaintextMatrixDimensions: dims, maxQueryCount: 1,
                encryptionParameters: parameters, scheme: Bfv<UInt64>.self),
            using: secretKey)
        let results = try await db.mulTranspose(vector: queryCt, using: key)
        for ct in results {
            let budget = try ct.noiseBudget(using: secretKey, variableTime: true)
            #expect(
                budget > Bfv<UInt64>.minNoiseBudget,
                """
                C×C kernel ran out of noise budget on 16×16 / logq=27,28,28: \
                got \(budget), need > \(Bfv<UInt64>.minNoiseBudget)
                """)
        }
    }

    @Test
    func mulVectorExactVsGroundTruth() async throws {
        let parameters = try EncryptionParameters<UInt64>(from: .n_4096_logq_27_28_28_logt_16)
        let context = try Bfv<UInt64>.Context(encryptionParameters: parameters)
        let secretKey = try context.generateSecretKey()
        let p = parameters.plaintextModulus

        let rowCount = 16
        let columnCount = 16
        let plaintextRows: [[UInt64]] = (0..<rowCount).map { r in
            (0..<columnCount).map { c in UInt64((r * 3 + c * 5) % 11) }
        }
        let queryValues: [UInt64] = (0..<columnCount).map { UInt64((7 &* $0 &+ 2) % 13) }

        // Ground truth: integer dot-products mod plaintextModulus.
        let expected: [UInt64] = plaintextRows.map { row in
            var acc: UInt64 = 0
            for (a, b) in zip(row, queryValues) {
                acc = (acc &+ (a &* b) % p) % p
            }
            return acc
        }

        let dims = try MatrixDimensions(rowCount: rowCount, columnCount: columnCount)
        let bsgs = BabyStepGiantStep(vectorDimension: columnCount)

        let db = try PlaintextMatrix<Bfv<UInt64>, Coeff>(
            context: context, dimensions: dims,
            packing: .diagonal(babyStepGiantStep: bsgs),
            values: plaintextRows.flatMap(\.self)).encrypt(using: secretKey)
        let queryDims = try MatrixDimensions(rowCount: 1, columnCount: columnCount)
        let queryCt = try PlaintextMatrix<Bfv<UInt64>, Coeff>(
            context: context, dimensions: queryDims,
            packing: .denseRow, values: queryValues).encrypt(using: secretKey)

        let key = try context.generateEvaluationKey(
            config: CTMatrixMultiplication.evaluationKeyConfig(
                plaintextMatrixDimensions: dims, maxQueryCount: 1,
                encryptionParameters: parameters, scheme: Bfv<UInt64>.self),
            using: secretKey)
        let results = try await db.mulTranspose(vector: queryCt, using: key)

        var decoded: [UInt64] = []
        for ct in results {
            decoded += try ct.decrypt(using: secretKey).decode(format: .simd) as [UInt64]
        }

        // Output is dense-column packed: first `rowCount` SIMD slots hold the scores.
        let scores = Array(decoded.prefix(rowCount))
        #expect(scores == expected, "decoded=\(scores) expected=\(expected)")
    }

    @Test
    func cosineSimilarityEndToEndPreservesScores() async throws {
        // Full pipeline: float vectors → normalize+scale+round → encrypt DB + query
        //   → CT kernel → decrypt → de-scale → compare to ground-truth cosine similarity.
        let parameters = try EncryptionParameters<UInt64>(from: .n_4096_logq_27_28_28_logt_16)
        let context = try Bfv<UInt64>.Context(encryptionParameters: parameters)
        let secretKey = try context.generateSecretKey()
        let p: UInt64 = parameters.plaintextModulus
        let scalingFactor: Float = 128

        let rowCount = 16
        let columnCount = 16

        // Deterministic pseudo-random float DB + query so the test is reproducible.
        var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
        func nextFloat() -> Float {
            seed &*= 0x100_0000_01B3
            seed ^= seed >> 33
            return Float(Int32(truncatingIfNeeded: seed)) / Float(Int32.max)
        }
        let dbData = (0..<rowCount * columnCount).map { _ in nextFloat() }
        let queryData = (0..<columnCount).map { _ in nextFloat() }
        let dbFloat = Array2d(data: dbData, rowCount: rowCount, columnCount: columnCount)
        let queryFloat = Array2d(data: queryData, rowCount: 1, columnCount: columnCount)

        // Ground truth cosine similarity: for each DB row, normalize both sides to unit L2
        // and compute the dot product. Expected range: [-1, 1].

        func l2normalize(_ v: [Float]) -> [Float] {
            let norm = v.map { $0 * $0 }.reduce(0, +).squareRoot()
            return norm.isZero ? v : v.map { $0 / norm }
        }
        let queryUnit = l2normalize(queryData)
        let expectedScores: [Float] = (0..<rowCount).map { r in
            let row = Array(dbData[r * columnCount..<(r + 1) * columnCount])
            let rowUnit = l2normalize(row)
            return zip(rowUnit, queryUnit).map(*).reduce(0, +)
        }

        // Matches the real client/server pipeline (ProcessedDatabase.swift:204, Client.swift:74).
        let dbScaled: Array2d<Int64> = dbFloat.normalizedScaledAndRounded(scalingFactor: scalingFactor)
        let queryScaled: Array2d<Int64> = queryFloat.normalizedScaledAndRounded(scalingFactor: scalingFactor)

        let dims = try MatrixDimensions(rowCount: rowCount, columnCount: columnCount)
        let bsgs = BabyStepGiantStep(vectorDimension: columnCount)

        let db = try PlaintextMatrix<Bfv<UInt64>, Coeff>(
            context: context,
            dimensions: dims,
            packing: .diagonal(babyStepGiantStep: bsgs),
            signedValues: dbScaled.data,
            reduce: false).encrypt(using: secretKey)

        let queryDims = try MatrixDimensions(rowCount: 1, columnCount: columnCount)
        let queryCt = try PlaintextMatrix<Bfv<UInt64>, Coeff>(
            context: context,
            dimensions: queryDims,
            packing: .denseRow,
            signedValues: queryScaled.data,
            reduce: false).encrypt(using: secretKey)

        let evalKey = try context.generateEvaluationKey(
            config: CTMatrixMultiplication.evaluationKeyConfig(
                plaintextMatrixDimensions: dims, maxQueryCount: 1,
                encryptionParameters: parameters, scheme: Bfv<UInt64>.self),
            using: secretKey)

        let results = try await db.mulTranspose(vector: queryCt, using: evalKey)

        var decoded: [UInt64] = []
        for ct in results {
            decoded += try ct.decrypt(using: secretKey).decode(format: .simd) as [UInt64]
        }

        // De-scale: centered-signed → divide by S² → float.
        let s2 = scalingFactor * scalingFactor
        let recovered: [Float] = decoded.prefix(rowCount).map { u in
            Float(u.remainderToCentered(modulus: p)) / s2
        }

        // Error sources: per-coord rounding error ≈ 1/(2S); cumulative RMS across d coords
        // bounded by √d / (2S). For d=16, S=128: ≈ 4/256 ≈ 0.016. Use 0.05 for headroom.
        let tolerance: Float = 0.05
        for (i, (got, want)) in zip(recovered, expectedScores).enumerated() {
            let err = abs(got - want)
            #expect(
                err < tolerance,
                "row \(i): recovered=\(got) expected=\(want) err=\(err) > \(tolerance)")
        }
    }

    @Test
    func cosineSimilarity192DimHighScalingPreservesScores() async throws {
        // Candidate "realistic" parameters for a 192-dim embedding with high scaling.
        //   N=8192, 4×40-bit coefficient moduli (logq≈160), logt=20 (p≈1M).
        //   Expected max |integer score| = S² ≤ 512² = 262144 << p ≈ 1_048_576, so
        //   there's no plaintext-modulus wrap-around for any S ≤ 512.
        //   Noise-budget-wise, four 40-bit primes give plenty of headroom for
        //   one C×C + relinearize + BSGS rotate-sum.
        let degree = 8192
        let coefficientModuli = try UInt64.generatePrimes(
            significantBitCounts: [40, 40, 40, 40],
            preferringSmall: false,
            nttDegree: degree)
        let plaintextModulus = try UInt64.generatePrimes(
            significantBitCounts: [20],
            preferringSmall: true,
            nttDegree: degree)[0]
        let parameters = try EncryptionParameters<UInt64>(
            polyDegree: degree,
            plaintextModulus: plaintextModulus,
            coefficientModuli: coefficientModuli,
            errorStdDev: .stdDev32,
            securityLevel: .unchecked)
        let context = try Bfv<UInt64>.Context(encryptionParameters: parameters)

        try await runCosineSimilarityParity(inputs: CosineSimilarityParityInputs(
            context: context, parameters: parameters, scalingFactor: 256,
            rowCount: 32, columnCount: 192, seed: 0x1234_5678_9ABC_DEF0))
        try await runCosineSimilarityParity(inputs: CosineSimilarityParityInputs(
            context: context, parameters: parameters, scalingFactor: 512,
            rowCount: 32, columnCount: 192, seed: 0xDEAD_BEEF_CAFE_BABE))
    }

    private func runCosineSimilarityParity(
        inputs: CosineSimilarityParityInputs) async throws
    {
        let context = inputs.context
        let parameters = inputs.parameters
        let scalingFactor = inputs.scalingFactor
        let rowCount = inputs.rowCount
        let columnCount = inputs.columnCount
        let initialSeed = inputs.seed
        let secretKey = try context.generateSecretKey()
        let p: UInt64 = parameters.plaintextModulus

        var seed = initialSeed
        func nextFloat() -> Float {
            seed &*= 0x100_0000_01B3
            seed ^= seed >> 33
            return Float(Int32(truncatingIfNeeded: seed)) / Float(Int32.max)
        }
        let dbData = (0..<rowCount * columnCount).map { _ in nextFloat() }
        let queryData = (0..<columnCount).map { _ in nextFloat() }
        let dbFloat = Array2d(data: dbData, rowCount: rowCount, columnCount: columnCount)
        let queryFloat = Array2d(data: queryData, rowCount: 1, columnCount: columnCount)

        func l2normalize(_ v: [Float]) -> [Float] {
            let norm = v.map { $0 * $0 }.reduce(0, +).squareRoot()
            return norm.isZero ? v : v.map { $0 / norm }
        }
        let queryUnit = l2normalize(queryData)
        let expectedScores: [Float] = (0..<rowCount).map { r in
            let row = Array(dbData[r * columnCount..<(r + 1) * columnCount])
            let rowUnit = l2normalize(row)
            return zip(rowUnit, queryUnit).map(*).reduce(0, +)
        }

        let dbScaled: Array2d<Int64> =
            dbFloat.normalizedScaledAndRounded(scalingFactor: scalingFactor)
        let queryScaled: Array2d<Int64> =
            queryFloat.normalizedScaledAndRounded(scalingFactor: scalingFactor)

        let dims = try MatrixDimensions(rowCount: rowCount, columnCount: columnCount)
        let bsgs = BabyStepGiantStep(vectorDimension: columnCount)

        let db = try PlaintextMatrix<Bfv<UInt64>, Coeff>(
            context: context, dimensions: dims,
            packing: .diagonal(babyStepGiantStep: bsgs),
            signedValues: dbScaled.data, reduce: false).encrypt(using: secretKey)

        let queryDims = try MatrixDimensions(rowCount: 1, columnCount: columnCount)
        let queryCt = try PlaintextMatrix<Bfv<UInt64>, Coeff>(
            context: context, dimensions: queryDims,
            packing: .denseRow,
            signedValues: queryScaled.data, reduce: false).encrypt(using: secretKey)

        let evalKey = try context.generateEvaluationKey(
            config: CTMatrixMultiplication.evaluationKeyConfig(
                plaintextMatrixDimensions: dims, maxQueryCount: 1,
                encryptionParameters: parameters, scheme: Bfv<UInt64>.self),
            using: secretKey)

        let results = try await db.mulTranspose(vector: queryCt, using: evalKey)

        // Observe + assert noise budget.
        let minBudget = try results.map { ciphertext in
            try ciphertext.noiseBudget(using: secretKey, variableTime: true)
        }.min() ?? -.infinity
        let logq = parameters.coefficientModuli.map(\.log2)
        print(
            "[d=\(columnCount) S=\(scalingFactor) N=\(parameters.polyDegree) " +
                "logq=\(logq)] noise budget = \(minBudget) bits")
        #expect(minBudget > Bfv<UInt64>.minNoiseBudget, "noise budget collapsed: \(minBudget)")

        var decoded: [UInt64] = []
        for ct in results {
            decoded += try ct.decrypt(using: secretKey).decode(format: .simd) as [UInt64]
        }
        let s2 = scalingFactor * scalingFactor
        let recovered: [Float] = decoded.prefix(rowCount).map { u in
            Float(u.remainderToCentered(modulus: p)) / s2
        }

        // Rounding-only bound: √d / (2S). d=192 S=256 → ≈0.027; S=512 → ≈0.0135.
        // Allow 4× headroom to be robust against L2-normalization numerical noise.
        let tolerance = 4 * Float(columnCount).squareRoot() / (2 * scalingFactor)
        var worst: Float = 0
        for (i, (got, want)) in zip(recovered, expectedScores).enumerated() {
            let err = abs(got - want)
            if err > worst { worst = err }
            #expect(
                err < tolerance,
                "row \(i) (S=\(scalingFactor)): recovered=\(got) expected=\(want) err=\(err) > \(tolerance)")
        }
        print("[d=\(columnCount) S=\(scalingFactor)] worst absolute error = \(worst), tolerance = \(tolerance)")
    }

    @Test
    func fullPipelineDatabaseToEncryptedScores() async throws {
        // End-to-end: Database (DatabaseRow's with entryId + metadata + float vector)
        //            + query vector
        //   → ClientConfig/ServerConfig (exactly as the real PNNS client/server use them)
        //   → QuantizedDatabase (same normalize+scale+round as Database.process)
        //   → diagonally-packed encrypted database + denseRow-packed encrypted query
        //   → CT kernel (mulTranspose) produces encrypted similarity scores
        //   → decrypt + centered-signed + divide by S² → Float cosine similarities
        //   → assert scores match ground-truth cosine similarity, and entryIds/metadata
        //     propagate through unchanged so the client can pair them back up.

        // --- Parameters ---
        let degree = 8192
        let coefficientModuli = try UInt64.generatePrimes(
            significantBitCounts: [40, 40, 40, 40],
            preferringSmall: false, nttDegree: degree)
        let plaintextModulus = try UInt64.generatePrimes(
            significantBitCounts: [20],
            preferringSmall: true, nttDegree: degree)[0]
        let parameters = try EncryptionParameters<UInt64>(
            polyDegree: degree,
            plaintextModulus: plaintextModulus,
            coefficientModuli: coefficientModuli,
            errorStdDev: .stdDev32,
            securityLevel: .unchecked)
        let context = try Bfv<UInt64>.Context(encryptionParameters: parameters)

        // --- Build a Database with IDs + metadata + float vectors ---
        let vectorDimension = 192
        let rowCount = 8
        var seed: UInt64 = 0xFEED_FACE_DEAD_BEEF
        func nextFloat() -> Float {
            seed &*= 0x100_0000_01B3
            seed ^= seed >> 33
            return Float(Int32(truncatingIfNeeded: seed)) / Float(Int32.max)
        }
        let rows: [DatabaseRow] = (0..<rowCount).map { i in
            DatabaseRow(
                entryId: UInt64(1000 + i),
                entryMetadata: [UInt8(i), UInt8(0xA0 &+ i)],
                vector: (0..<vectorDimension).map { _ in nextFloat() })
        }
        let database = Database(rows: rows)
        let queryVector: [Float] = (0..<vectorDimension).map { _ in nextFloat() }

        // --- Build the client & server configs (CT path uses our evaluationKeyConfig) ---
        let scalingFactor = 512
        let maxSafeS = ClientConfig<Bfv<UInt64>>.maxScalingFactor(
            distanceMetric: .cosineSimilarity,
            vectorDimension: vectorDimension,
            plaintextModuli: [plaintextModulus])
        #expect(scalingFactor <= maxSafeS,
                "S=\(scalingFactor) exceeds maxScalingFactor=\(maxSafeS) for this plaintext modulus")

        let dbDims = try MatrixDimensions(rowCount: rowCount, columnCount: vectorDimension)
        let bsgs = BabyStepGiantStep(vectorDimension: vectorDimension)
        let evalConfig = try CTMatrixMultiplication.evaluationKeyConfig(
            plaintextMatrixDimensions: dbDims, maxQueryCount: 1,
            encryptionParameters: parameters, scheme: Bfv<UInt64>.self)
        let clientConfig = try ClientConfig<Bfv<UInt64>>(
            encryptionParameters: parameters,
            scalingFactor: scalingFactor,
            queryPacking: .denseRow,
            vectorDimension: vectorDimension,
            evaluationKeyConfig: evalConfig,
            distanceMetric: .cosineSimilarity)
        let serverConfig = ServerConfig(
            clientConfig: clientConfig,
            databasePacking: .diagonal(babyStepGiantStep: bsgs))

        // --- Client: generate secret & evaluation keys ---
        let secretKey = try context.generateSecretKey()
        let evaluationKey = try context.generateEvaluationKey(config: evalConfig, using: secretKey)

        // --- Quantize the database (same step Database.process does internally) ---
        let quantized = QuantizedDatabase(database: database, config: serverConfig)
        #expect(quantized.entryIds == rows.map(\.entryId))
        #expect(quantized.entryMetadatas == rows.map(\.entryMetadata))

        // --- Client: produce the encrypted database (diagonal / BSGS packed) ---
        let dbPlaintext = try PlaintextMatrix<Bfv<UInt64>, Coeff>(
            context: context,
            dimensions: dbDims,
            packing: serverConfig.databasePacking,
            signedValues: quantized.signedValues,
            reduce: false)
        let encryptedDb = try dbPlaintext.encrypt(using: secretKey)

        // --- Client: produce the encrypted query (denseRow packed) ---
        let queryFloatMatrix = Array2d(data: queryVector, rowCount: 1, columnCount: vectorDimension)
        let queryScaled: Array2d<Int64> =
            queryFloatMatrix.normalizedScaledAndRounded(scalingFactor: Float(scalingFactor))
        let queryDims = try MatrixDimensions(rowCount: 1, columnCount: vectorDimension)
        let queryPlaintext = try PlaintextMatrix<Bfv<UInt64>, Coeff>(
            context: context,
            dimensions: queryDims,
            packing: clientConfig.queryPacking,
            signedValues: queryScaled.data,
            reduce: false)
        let encryptedQuery = try queryPlaintext.encrypt(using: secretKey)

        // --- Offshore server: compute encrypted similarity scores. ---
        // The server holds: encryptedDb, the evaluationKey, entryIds, entryMetadatas.
        // The server does NOT hold the secret key.
        let responseCiphertexts = try await encryptedDb.mulTranspose(
            vector: encryptedQuery, using: evaluationKey)

        // Observe noise budget on the way out so we know we didn't overspend.
        let budget = try responseCiphertexts.map { ciphertext in
            try ciphertext.noiseBudget(using: secretKey, variableTime: true)
        }.min() ?? -.infinity
        print("[full-pipeline d=\(vectorDimension) S=\(scalingFactor)] noise budget = \(budget) bits")
        #expect(budget > Bfv<UInt64>.minNoiseBudget, "noise budget collapsed: \(budget)")

        // --- Client: decrypt, un-CRT (single modulus = identity), de-scale ---
        var decoded: [UInt64] = []
        for ct in responseCiphertexts {
            decoded += try ct.decrypt(using: secretKey).decode(format: .simd) as [UInt64]
        }
        let s2 = Float(scalingFactor * scalingFactor)
        let recoveredScores: [Float] = decoded.prefix(rowCount).map { u in
            Float(u.remainderToCentered(modulus: plaintextModulus)) / s2
        }

        // --- Ground truth: cosine similarity of the unit vectors ---

        func l2normalize(_ v: [Float]) -> [Float] {
            let norm = v.map { $0 * $0 }.reduce(0, +).squareRoot()
            return norm.isZero ? v : v.map { $0 / norm }
        }
        let queryUnit = l2normalize(queryVector)
        let expectedScores: [Float] = rows.map { row in
            let unit = l2normalize(row.vector)
            return zip(unit, queryUnit).map(*).reduce(0, +)
        }

        // Rounding-only analytical bound: √d / (2S) ≈ 0.0135 for d=192, S=512.
        // 4× headroom for accumulated float-normalization noise in the ground-truth side.
        let tolerance = 4 * Float(vectorDimension).squareRoot() / (2 * Float(scalingFactor))

        // --- Final assertions: scores match, entries map back by id/metadata ---
        for i in 0..<rowCount {
            let err = abs(recoveredScores[i] - expectedScores[i])
            let message = """
                id=\(rows[i].entryId) meta=\(rows[i].entryMetadata) \
                got=\(recoveredScores[i]) want=\(expectedScores[i]) err=\(err) > \(tolerance)
                """
            #expect(err < tolerance, Comment(rawValue: message))
        }

        // And the server-side plumbing (entryIds, metadata) survives the round-trip.
        // (In a future Server/Client CT API, these would ride alongside the response
        // so the client can pair (entryId, score) tuples.)
        for i in 0..<rowCount {
            #expect(quantized.entryIds[i] == rows[i].entryId)
            #expect(quantized.entryMetadatas[i] == rows[i].entryMetadata)
        }
    }

    @Test
    func txtpbDatabaseToEncryptedBinpbRoundTrip() async throws {
        // End-to-end file-format pipeline:
        //   1. Build a Database in memory and write it as `.txtpb` (same format
        //      PNNSProcessDatabase consumes).
        //   2. Read the `.txtpb` back → native Database (same code path as the
        //      real CLI: Apple_..._Database(from: path).native()).
        //   3. Quantize + diagonally-pack + encrypt under a secret key.
        //   4. Serialize the encrypted database as a `.binpb` via
        //      SerializedCiphertextMatrix → protobuf → save().
        //   5. Load the `.binpb` from disk, deserialize → CiphertextMatrix.
        //   6. Encrypt a query under the same secret key, run the CT kernel on
        //      the ship-and-load-from-disk encrypted DB, decrypt, de-scale.
        //   7. Assert the recovered scores match ground-truth cosine similarity.
        //
        // This proves the CT path can feed from `.txtpb` database files
        // (the existing input format) and emit `.binpb` blobs that an offshore
        // server can mmap and evaluate queries against, without the secret key.

        let fm = FileManager.default
        let tmpDir = fm.temporaryDirectory.appendingPathComponent("ctpnns-\(UUID().uuidString)")
        try fm.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tmpDir) }
        let txtpbPath = tmpDir.appendingPathComponent("database.txtpb").path
        let binpbPath = tmpDir.appendingPathComponent("encrypted-database.binpb").path

        // --- 1. Build Database in memory ---
        let vectorDimension = 64
        let rowCount = 4
        var seed: UInt64 = 0xABCD_1234_5678_9ABC
        func nextFloat() -> Float {
            seed &*= 0x100_0000_01B3
            seed ^= seed >> 33
            return Float(Int32(truncatingIfNeeded: seed)) / Float(Int32.max)
        }
        let originalRows: [DatabaseRow] = (0..<rowCount).map { i in
            DatabaseRow(
                entryId: UInt64(42000 + i),
                entryMetadata: [UInt8(0xE0 &+ i), UInt8(i)],
                vector: (0..<vectorDimension).map { _ in nextFloat() })
        }
        let queryVector: [Float] = (0..<vectorDimension).map { _ in nextFloat() }
        let originalDatabase = Database(rows: originalRows)

        // --- 2a. Write Database to .txtpb ---
        try originalDatabase.proto().save(to: txtpbPath)

        // --- 2b. Read .txtpb back into native Database (mirrors PNNSProcessDatabase) ---
        let loadedDbProto = try Apple_SwiftHomomorphicEncryption_Pnns_V1_Database(from: txtpbPath)
        let database: Database = loadedDbProto.native()
        #expect(database.rows.count == rowCount)
        for (i, row) in database.rows.enumerated() {
            #expect(row.entryId == originalRows[i].entryId)
            #expect(row.entryMetadata == originalRows[i].entryMetadata)
            #expect(row.vector == originalRows[i].vector)
        }

        // --- 3. Encryption parameters, configs, secret key ---
        let degree = 8192
        let coefficientModuli = try UInt64.generatePrimes(
            significantBitCounts: [40, 40, 40, 40],
            preferringSmall: false, nttDegree: degree)
        let plaintextModulus = try UInt64.generatePrimes(
            significantBitCounts: [20], preferringSmall: true, nttDegree: degree)[0]
        let parameters = try EncryptionParameters<UInt64>(
            polyDegree: degree,
            plaintextModulus: plaintextModulus,
            coefficientModuli: coefficientModuli,
            errorStdDev: .stdDev32,
            securityLevel: .unchecked)
        let context = try Bfv<UInt64>.Context(encryptionParameters: parameters)
        let secretKey = try context.generateSecretKey()

        let scalingFactor = 512
        let dbDims = try MatrixDimensions(rowCount: rowCount, columnCount: vectorDimension)
        let bsgs = BabyStepGiantStep(vectorDimension: vectorDimension)
        let evalConfig = try CTMatrixMultiplication.evaluationKeyConfig(
            plaintextMatrixDimensions: dbDims, maxQueryCount: 1,
            encryptionParameters: parameters, scheme: Bfv<UInt64>.self)
        let clientConfig = try ClientConfig<Bfv<UInt64>>(
            encryptionParameters: parameters,
            scalingFactor: scalingFactor,
            queryPacking: .denseRow,
            vectorDimension: vectorDimension,
            evaluationKeyConfig: evalConfig,
            distanceMetric: .cosineSimilarity)
        let serverConfig = ServerConfig(
            clientConfig: clientConfig,
            databasePacking: .diagonal(babyStepGiantStep: bsgs))
        let evalKey = try context.generateEvaluationKey(config: evalConfig, using: secretKey)

        // --- 4a. Quantize + pack + encrypt DB ---
        let quantized = QuantizedDatabase(database: database, config: serverConfig)
        let dbPlaintext = try PlaintextMatrix<Bfv<UInt64>, Coeff>(
            context: context,
            dimensions: dbDims,
            packing: serverConfig.databasePacking,
            signedValues: quantized.signedValues,
            reduce: false)
        let encryptedDb = try dbPlaintext.encrypt(using: secretKey)

        // --- 4b. Serialize encrypted DB → SerializedCiphertextMatrix → proto → .binpb ---
        let serializedCtMatrix: SerializedCiphertextMatrix<UInt64> = try encryptedDb.serialize()
        let protoCtMatrix = try serializedCtMatrix.proto()
        try protoCtMatrix.save(to: binpbPath)

        // Report the on-disk size so the user can calibrate expectations.
        let binSize = try fm.attributesOfItem(atPath: binpbPath)[.size] as? Int ?? -1
        print("[txtpb→binpb] encrypted-database.binpb on disk = \(binSize) bytes " +
            "(rows=\(rowCount), dim=\(vectorDimension), N=\(degree))")

        // --- 5. Read .binpb back → SerializedCiphertextMatrix → CiphertextMatrix ---
        let loadedProto = try Apple_SwiftHomomorphicEncryption_Pnns_V1_SerializedCiphertextMatrix(
            from: binpbPath)
        let loadedSerialized: SerializedCiphertextMatrix<UInt64> = try loadedProto.native()
        let loadedEncryptedDb = try CiphertextMatrix<Bfv<UInt64>, Coeff>(
            deserialize: loadedSerialized, context: context)
        #expect(loadedEncryptedDb.ciphertexts.count == encryptedDb.ciphertexts.count)

        // --- 6. Encrypt query under the same secret key ---
        let queryFloat = Array2d(data: queryVector, rowCount: 1, columnCount: vectorDimension)
        let queryScaled: Array2d<Int64> =
            queryFloat.normalizedScaledAndRounded(scalingFactor: Float(scalingFactor))
        let queryDims = try MatrixDimensions(rowCount: 1, columnCount: vectorDimension)
        let queryPlaintext = try PlaintextMatrix<Bfv<UInt64>, Coeff>(
            context: context,
            dimensions: queryDims,
            packing: clientConfig.queryPacking,
            signedValues: queryScaled.data,
            reduce: false)
        let encryptedQuery = try queryPlaintext.encrypt(using: secretKey)

        // --- 7. Offshore server runs the kernel on the loaded-from-disk encrypted DB ---
        let responseCts = try await loadedEncryptedDb.mulTranspose(
            vector: encryptedQuery, using: evalKey)

        let budget = try responseCts.map { ciphertext in
            try ciphertext.noiseBudget(using: secretKey, variableTime: true)
        }.min() ?? -.infinity
        #expect(budget > Bfv<UInt64>.minNoiseBudget, "noise budget collapsed: \(budget)")

        // --- 8. Client decrypts + de-scales ---
        var decoded: [UInt64] = []
        for ct in responseCts {
            decoded += try ct.decrypt(using: secretKey).decode(format: .simd) as [UInt64]
        }
        let s2 = Float(scalingFactor * scalingFactor)
        let recoveredScores: [Float] = decoded.prefix(rowCount).map { u in
            Float(u.remainderToCentered(modulus: plaintextModulus)) / s2
        }

        // --- 9. Ground truth + assertions ---

        func l2normalize(_ v: [Float]) -> [Float] {
            let norm = v.map { $0 * $0 }.reduce(0, +).squareRoot()
            return norm.isZero ? v : v.map { $0 / norm }
        }
        let queryUnit = l2normalize(queryVector)
        let expectedScores: [Float] = originalRows.map { row in
            let unit = l2normalize(row.vector)
            return zip(unit, queryUnit).map(*).reduce(0, +)
        }
        let tolerance = 4 * Float(vectorDimension).squareRoot() / (2 * Float(scalingFactor))
        for i in 0..<rowCount {
            let err = abs(recoveredScores[i] - expectedScores[i])
            let msg = """
                entryId=\(originalRows[i].entryId) got=\(recoveredScores[i]) \
                want=\(expectedScores[i]) err=\(err) tol=\(tolerance)
                """
            #expect(err < tolerance, Comment(rawValue: msg))
        }
    }

    @Test
    func batchMulTransposeMatchesSingleQueries() async throws {
        // Batched query (N rows) via mulTranspose(matrix:) must equal N independent
        // mulTranspose(vector:) calls, one per row.
        //
        // The batch kernel does strictly more ciphertext ops than the vector kernel
        // (extractDenseRow + final rotate-and-sum repacking), so the 3×28-bit preset is
        // too tight here — we use 4×40-bit/N=8192 (same as the end-to-end tests) for
        // comfortable noise-budget headroom.
        let degree = 8192
        let coefficientModuli = try UInt64.generatePrimes(
            significantBitCounts: [40, 40, 40, 40],
            preferringSmall: false, nttDegree: degree)
        let plaintextModulus = try UInt64.generatePrimes(
            significantBitCounts: [20], preferringSmall: true, nttDegree: degree)[0]
        let parameters = try EncryptionParameters<UInt64>(
            polyDegree: degree,
            plaintextModulus: plaintextModulus,
            coefficientModuli: coefficientModuli,
            errorStdDev: .stdDev32,
            securityLevel: .unchecked)
        let context = try Bfv<UInt64>.Context(encryptionParameters: parameters)
        let secretKey = try context.generateSecretKey()

        let dbRowCount = 8
        let columnCount = 16
        let queryRowCount = 4
        let plaintextRows: [[UInt64]] = (0..<dbRowCount).map { r in
            (0..<columnCount).map { c in UInt64((r &+ 3 &* c) % 5) }
        }
        let queryRows: [[UInt64]] = (0..<queryRowCount).map { q in
            (0..<columnCount).map { c in UInt64((q &+ 2 &* c &+ 1) % 4) }
        }

        let dbDims = try MatrixDimensions(rowCount: dbRowCount, columnCount: columnCount)
        let bsgs = BabyStepGiantStep(vectorDimension: columnCount)
        let db = try PlaintextMatrix<Bfv<UInt64>, Coeff>(
            context: context, dimensions: dbDims,
            packing: .diagonal(babyStepGiantStep: bsgs),
            values: plaintextRows.flatMap(\.self)).encrypt(using: secretKey)

        let key = try context.generateEvaluationKey(
            config: CTMatrixMultiplication.evaluationKeyConfig(
                plaintextMatrixDimensions: dbDims,
                maxQueryCount: queryRowCount,
                encryptionParameters: parameters, scheme: Bfv<UInt64>.self),
            using: secretKey)

        // Independent reference: run one single-row mulTranspose(vector:) per query row.
        var perRowDecoded: [[UInt64]] = []
        for q in queryRows {
            let qDims = try MatrixDimensions(rowCount: 1, columnCount: columnCount)
            let qCt = try PlaintextMatrix<Bfv<UInt64>, Coeff>(
                context: context, dimensions: qDims,
                packing: .denseRow, values: q).encrypt(using: secretKey)
            let cts = try await db.mulTranspose(vector: qCt, using: key)
            var decoded: [UInt64] = []
            for ct in cts {
                decoded += try ct.decrypt(using: secretKey).decode(format: .simd) as [UInt64]
            }
            perRowDecoded.append(Array(decoded.prefix(dbRowCount)))
        }

        // Batched: single mulTranspose(matrix:) with all queries packed.
        let queryBatchDims = try MatrixDimensions(
            rowCount: queryRowCount, columnCount: columnCount)
        let queryBatchCt = try PlaintextMatrix<Bfv<UInt64>, Coeff>(
            context: context, dimensions: queryBatchDims,
            packing: .denseRow,
            values: queryRows.flatMap(\.self)).encrypt(using: secretKey)
        let resultMatrix = try await db.mulTranspose(matrix: queryBatchCt, using: key)

        // Decode the dense-column-packed result and split into per-query columns.
        var flatDecoded: [UInt64] = []
        for ct in resultMatrix.ciphertexts {
            flatDecoded += try ct.decrypt(using: secretKey).decode(format: .simd) as [UInt64]
        }
        // Dense-column layout: for each result ciphertext, slot values hold up to
        // `columnsPerSimdRowCount` columns interleaved by `dimensions.rowCount`.
        // Easiest path: assume small enough to fit in one ciphertext, and confirm each
        // per-query column matches the reference. The plaintext kernel uses the same
        // packing, so we reuse the layout logic by taking the first `dbRowCount` slots
        // of each column's stride.
        let p = parameters.plaintextModulus
        for q in 0..<queryRowCount {
            let reference = perRowDecoded[q]
            // Expected integer dot product mod p, matching the single-vector path.
            let expected: [UInt64] = (0..<dbRowCount).map { r in
                var acc: UInt64 = 0
                for (a, b) in zip(plaintextRows[r], queryRows[q]) {
                    acc = (acc &+ (a &* b) % p) % p
                }
                return acc
            }
            #expect(reference == expected, "reference diverged for query row \(q)")
        }

        // Decode the single batch ciphertext directly and split by stride = dbRowCount.
        // Dense-column packing puts query q's scores in slots [q*dbRowCount .. (q+1)*dbRowCount).
        var batchDecoded: [UInt64] = []
        for ct in resultMatrix.ciphertexts {
            batchDecoded += try ct.decrypt(using: secretKey).decode(format: .simd) as [UInt64]
        }
        for q in 0..<queryRowCount {
            let base = q * dbRowCount
            let slice = Array(batchDecoded[base..<base + dbRowCount])
            #expect(slice == perRowDecoded[q],
                    "batch[query=\(q)] got=\(slice) want=\(perRowDecoded[q])")
        }
    }

    @Test
    func timingReport() async throws {
        // Print wall-clock timing of the CT kernel at a few scales. Not an assertion —
        // purely informational. Run with `swift test -c release -Xswiftc -O` for
        // realistic numbers (debug mode is ~5-10× slower).
        let degree = 8192
        let coefficientModuli = try UInt64.generatePrimes(
            significantBitCounts: [40, 40, 40, 40],
            preferringSmall: false, nttDegree: degree)
        let plaintextModulus = try UInt64.generatePrimes(
            significantBitCounts: [20], preferringSmall: true, nttDegree: degree)[0]
        let parameters = try EncryptionParameters<UInt64>(
            polyDegree: degree,
            plaintextModulus: plaintextModulus,
            coefficientModuli: coefficientModuli,
            errorStdDev: .stdDev32,
            securityLevel: .unchecked)
        let context = try Bfv<UInt64>.Context(encryptionParameters: parameters)
        let secretKey = try context.generateSecretKey()

        struct TimingCase {
            let dim: Int
            let rows: Int
            let queries: Int
        }
        let cases: [TimingCase] = [
            TimingCase(dim: 64, rows: 32, queries: 1),
            TimingCase(dim: 128, rows: 32, queries: 1),
            TimingCase(dim: 192, rows: 32, queries: 1),
            TimingCase(dim: 192, rows: 32, queries: 4), // batched
            TimingCase(dim: 192, rows: 128, queries: 1),
            TimingCase(dim: 192, rows: 10000, queries: 4), // realistic: 10k-row DB, batched 4-query
        ]

        print("=== CTPrivateNearestNeighbourSearch kernel timing ===")
        print("    d  rows queries | setup_s kernel_s per_q_ms | noise(b)  mode")

        for tc in cases {
            var seed = UInt64(tc.dim) &* 0x9E37_79B9_7F4A_7C15
            func nextInt() -> UInt64 {
                seed &*= 0x100_0000_01B3
                seed ^= seed >> 33
                return seed % 7
            }
            let dbValues = (0..<tc.rows * tc.dim).map { _ in nextInt() }
            let qValues = (0..<tc.queries * tc.dim).map { _ in nextInt() }

            let dbDims = try MatrixDimensions(rowCount: tc.rows, columnCount: tc.dim)
            let bsgs = BabyStepGiantStep(vectorDimension: tc.dim)
            let queryDims = try MatrixDimensions(rowCount: tc.queries, columnCount: tc.dim)

            let clock = ContinuousClock()
            let setup = try clock.measure {
                _ = try PlaintextMatrix<Bfv<UInt64>, Coeff>(
                    context: context, dimensions: dbDims,
                    packing: .diagonal(babyStepGiantStep: bsgs),
                    values: dbValues).encrypt(using: secretKey)
            }
            let db = try PlaintextMatrix<Bfv<UInt64>, Coeff>(
                context: context, dimensions: dbDims,
                packing: .diagonal(babyStepGiantStep: bsgs),
                values: dbValues).encrypt(using: secretKey)
            let queryCt = try PlaintextMatrix<Bfv<UInt64>, Coeff>(
                context: context, dimensions: queryDims,
                packing: .denseRow, values: qValues).encrypt(using: secretKey)
            let evalKey = try context.generateEvaluationKey(
                config: CTMatrixMultiplication.evaluationKeyConfig(
                    plaintextMatrixDimensions: dbDims,
                    maxQueryCount: tc.queries,
                    encryptionParameters: parameters, scheme: Bfv<UInt64>.self),
                using: secretKey)

            let mode: String
            var minBudget: Double = 0
            var kernel: Duration = .zero
            if tc.queries == 1 {
                mode = "vector"
                let start = clock.now
                let results = try await db.mulTranspose(vector: queryCt, using: evalKey)
                kernel = clock.now - start
                minBudget = try results.map { ciphertext in
                    try ciphertext.noiseBudget(using: secretKey, variableTime: true)
                }.min() ?? 0
            } else {
                mode = "matrix"
                let start = clock.now
                let result = try await db.mulTranspose(matrix: queryCt, using: evalKey)
                kernel = clock.now - start
                minBudget = try result.ciphertexts.map { ciphertext in
                    try ciphertext.noiseBudget(using: secretKey, variableTime: true)
                }.min() ?? 0
            }

            let setupSec = Double(setup.components.seconds) +
                Double(setup.components.attoseconds) / 1e18
            let kernelSec = Double(kernel.components.seconds) +
                Double(kernel.components.attoseconds) / 1e18
            let perQueryMs = (kernelSec / Double(tc.queries)) * 1000
            let line = String(
                format: "%5d %5d %7d | %7.3f %8.3f %8.1f | %8.2f  %@",
                tc.dim, tc.rows, tc.queries,
                setupSec, kernelSec, perQueryMs,
                minBudget, mode as NSString)
            print(line)
        }
    }

    @Test
    func serverWrapperEndToEnd() async throws {
        let degree = 8192
        let coefficientModuli = try UInt64.generatePrimes(
            significantBitCounts: [40, 40, 40, 40],
            preferringSmall: false, nttDegree: degree)
        let plaintextModulus = try UInt64.generatePrimes(
            significantBitCounts: [20], preferringSmall: true, nttDegree: degree)[0]
        let parameters = try EncryptionParameters<UInt64>(
            polyDegree: degree,
            plaintextModulus: plaintextModulus,
            coefficientModuli: coefficientModuli,
            errorStdDev: .stdDev32,
            securityLevel: .unchecked)
        let context = try Bfv<UInt64>.Context(encryptionParameters: parameters)

        let vectorDimension = 128
        let rowCount = 6
        let queryRowCount = 2
        var seed: UInt64 = 0x1357_9BDF_2468_ACE0
        func nextFloat() -> Float {
            seed &*= 0x100_0000_01B3
            seed ^= seed >> 33
            return Float(Int32(truncatingIfNeeded: seed)) / Float(Int32.max)
        }
        let rows: [DatabaseRow] = (0..<rowCount).map { i in
            DatabaseRow(
                entryId: UInt64(9000 + i),
                entryMetadata: [UInt8(0xA0 &+ i), UInt8(i)],
                vector: (0..<vectorDimension).map { _ in nextFloat() })
        }
        let database = Database(rows: rows)
        let queryVectors: [[Float]] = (0..<queryRowCount).map { _ in
            (0..<vectorDimension).map { _ in nextFloat() }
        }

        let scalingFactor = 512
        let dbDims = try MatrixDimensions(rowCount: rowCount, columnCount: vectorDimension)
        let bsgs = BabyStepGiantStep(vectorDimension: vectorDimension)
        let evalConfig = try CTMatrixMultiplication.evaluationKeyConfig(
            plaintextMatrixDimensions: dbDims, maxQueryCount: queryRowCount,
            encryptionParameters: parameters, scheme: Bfv<UInt64>.self)
        let clientConfig = try ClientConfig<Bfv<UInt64>>(
            encryptionParameters: parameters,
            scalingFactor: scalingFactor,
            queryPacking: .denseRow,
            vectorDimension: vectorDimension,
            evaluationKeyConfig: evalConfig,
            distanceMetric: .cosineSimilarity)
        let serverConfig = ServerConfig(
            clientConfig: clientConfig,
            databasePacking: .diagonal(babyStepGiantStep: bsgs))

        // Client side: holds secret key, emits encrypted DB + query + eval key.
        let client = try Client<Bfv<UInt64>>(config: clientConfig, contexts: [context])
        let secretKey = try client.generateSecretKey()
        let evaluationKey = try client.generateEvaluationKey(using: secretKey)
        let encryptedDb = try database.processAndEncrypt(
            config: serverConfig, secretKey: secretKey, contexts: [context])
        #expect(encryptedDb.entryIds == rows.map(\.entryId))
        #expect(encryptedDb.entryMetadatas == rows.map(\.entryMetadata))

        let queryArray = Array2d(
            data: queryVectors.flatMap(\.self),
            rowCount: queryRowCount, columnCount: vectorDimension)
        let query = try client.generateQuery(for: queryArray, using: secretKey)

        // Offshore server holds encryptedDb + evaluationKey only; no secret key.
        let server = try CTPrivateNearestNeighbourSearch.Server(database: encryptedDb)
        let response = try await server.computeResponse(to: query, using: evaluationKey)
        #expect(response.entryIds == rows.map(\.entryId))
        #expect(response.entryMetadatas == rows.map(\.entryMetadata))

        let budget = try response.noiseBudget(using: secretKey, variableTime: true)
        print("[server-wrapper end-to-end] response noise budget = \(budget) bits")
        #expect(budget > Bfv<UInt64>.minNoiseBudget, "noise budget collapsed: \(budget)")

        // Decrypt via the existing plaintext-path Client — proves Response shape is compatible.
        let distances = try client.decrypt(response: response, using: secretKey)
        #expect(distances.entryIds == rows.map(\.entryId))
        #expect(distances.distances.rowCount == rowCount)
        #expect(distances.distances.columnCount == queryRowCount)

        func l2normalize(_ v: [Float]) -> [Float] {
            let norm = v.map { $0 * $0 }.reduce(0, +).squareRoot()
            return norm.isZero ? v : v.map { $0 / norm }
        }
        let tolerance = 4 * Float(vectorDimension).squareRoot() / (2 * Float(scalingFactor))
        for r in 0..<rowCount {
            let rowUnit = l2normalize(rows[r].vector)
            for q in 0..<queryRowCount {
                let qUnit = l2normalize(queryVectors[q])
                let expected = zip(rowUnit, qUnit).map(*).reduce(0, +)
                let got = distances.distances[r, q]
                let err = abs(got - expected)
                let message = "row=\(r) query=\(q) got=\(got) want=\(expected) err=\(err)"
                #expect(err < tolerance, Comment(rawValue: message))
            }
        }
    }
}
