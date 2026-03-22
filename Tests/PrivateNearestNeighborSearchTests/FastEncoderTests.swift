// Copyright 2025-2026 Apple Inc. and the Swift Homomorphic Encryption project authors
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

import Foundation
import HomomorphicEncryption
@testable import PrivateNearestNeighborSearch
import Testing

struct FastEncoderTests {
    private typealias Scheme = Bfv<UInt64>

    // Concrete parameters from server-config.txtpb
    private static let polyDegree = 8192
    private static let plaintextModulus: UInt64 = 536_903_681
    private static let coefficientModuli: [UInt64] = [
        36_028_797_018_652_673,
        36_028_797_017_571_329,
        36_028_797_017_456_641,
    ]
    private static let vectorDimension = 128
    private static let rowCount = 128

    @Test
    func fastEncoderMatchesStandard() async throws {
        try await runFastEncoderTest()
    }

    func runFastEncoderTest() async throws {
        let vectorDimension = Self.vectorDimension
        let rowCount = Self.rowCount

        let encryptionParameters = try EncryptionParameters<UInt64>(
            polyDegree: Self.polyDegree,
            plaintextModulus: Self.plaintextModulus,
            coefficientModuli: Self.coefficientModuli,
            errorStdDev: .stdDev32,
            securityLevel: .unchecked)

        let plaintextModuli: [UInt64] = [Self.plaintextModulus]
        let scalingFactor = ClientConfig<Scheme>.maxScalingFactor(
            distanceMetric: .dotProduct,
            vectorDimension: vectorDimension,
            plaintextModuli: plaintextModuli)

        let evaluationKeyConfig = try MatrixMultiplication.evaluationKeyConfig(
            plaintextMatrixDimensions: MatrixDimensions(
                rowCount: rowCount, columnCount: vectorDimension),
            maxQueryCount: 1,
            encryptionParameters: encryptionParameters,
            scheme: Scheme.self)

        let clientConfig = try ClientConfig<Scheme>(
            encryptionParameters: encryptionParameters,
            scalingFactor: scalingFactor,
            queryPacking: .denseRow,
            vectorDimension: vectorDimension,
            evaluationKeyConfig: evaluationKeyConfig,
            distanceMetric: .dotProduct)
        let serverConfig = ServerConfig(
            clientConfig: clientConfig,
            databasePacking: .diagonal(
                babyStepGiantStep: BabyStepGiantStep(vectorDimension: vectorDimension)))

        let context = try Scheme.Context(encryptionParameters: encryptionParameters)

        // Generate test vectors
        let rawVectors: [[Float]] = (0..<rowCount).map { rowIndex in
            (0..<vectorDimension).map { colIndex in
                let norm = (0..<vectorDimension).map { c in
                    Float(c + rowIndex) * Float(c + rowIndex)
                }.reduce(0, +).squareRoot()
                return Float(colIndex + rowIndex) * (rowIndex.isMultiple(of: 2) ? 1 : -1) / max(norm, 1)
            }
        }

        // Quantize
        let sf = Float(scalingFactor)
        let signedValues: [Scheme.SignedScalar] = rawVectors.flatMap { row in
            row.map { Scheme.SignedScalar(($0 * sf).rounded()) }
        }

        // === Standard pipeline ===
        let database = Database(rows: rawVectors.enumerated().map { i, vec in
            DatabaseRow(entryId: UInt64(i), entryMetadata: [], vector: vec)
        })

        let standardProcessed = try await database.process(config: serverConfig, contexts: [context])
        let standardMatrices = standardProcessed.plaintextMatrices

        // === Fast encoder ===

        let encoder = try FastPlaintextEncoder<Scheme>(config: serverConfig, context: context, rowCount: rowCount)

        let fastMatrix = try encoder.encode(signedValues: signedValues, context: context)

        // === Compare: decrypt both and check values match ===
        // The PlaintextMatrix plaintexts should produce identical results when
        // used in a query. Let's do a full round-trip comparison.
        let client = try Client<Scheme>(config: clientConfig, contexts: [context])
        let secretKey = try client.generateSecretKey()
        let evalKey = try client.generateEvaluationKey(using: secretKey)

        let queryRow = rawVectors[0].map(\.self) // query = first row
        let queryVectors = Array2d(data: [queryRow])
        let query = try client.generateQuery(for: queryVectors, using: secretKey)

        // Standard pipeline response
        let standardServer = try Server(database: standardProcessed)
        let standardResponse = try await standardServer.computeResponse(to: query, using: evalKey)
        let standardDistances = try client.decrypt(response: standardResponse, using: secretKey)

        // Fast encoder response
        let fastProcessed = try ProcessedDatabase<Scheme>(
            contexts: [context],
            plaintextMatrices: [fastMatrix],
            entryIds: (0..<UInt64(rowCount)).map(\.self),
            entryMetadatas: [],
            serverConfig: serverConfig)
        let fastServer = try Server(database: fastProcessed)
        let fastResponse = try await fastServer.computeResponse(to: query, using: evalKey)
        let fastDistances = try client.decrypt(response: fastResponse, using: secretKey)

        // Compare distances
        #expect(standardDistances.distances.data.count == fastDistances.distances.data.count)
        var maxDiff: Float = 0
        for i in 0..<standardDistances.distances.data.count {
            let diff = abs(standardDistances.distances.data[i] - fastDistances.distances.data[i])
            maxDiff = max(maxDiff, diff)
        }
        // They should be identical (same integer values, same encoding)
        #expect(maxDiff == 0, Comment(rawValue: "Fast encoder differs from standard: maxDiff=\(maxDiff)"))
    }

    /// Validate the full ORAM-friendly flow:
    /// Database → QuantizedDatabase → serialize → deserialize → FastPlaintextEncoder → query
    @Test
    func quantizedDatabaseRoundTrip() async throws {
        try await runQuantizedRoundTrip()
    }

    func runQuantizedRoundTrip() async throws {
        let vectorDimension = Self.vectorDimension
        let rowCount = Self.rowCount

        let encryptionParameters = try EncryptionParameters<UInt64>(
            polyDegree: Self.polyDegree,
            plaintextModulus: Self.plaintextModulus,
            coefficientModuli: Self.coefficientModuli,
            errorStdDev: .stdDev32,
            securityLevel: .unchecked)

        let plaintextModuli: [UInt64] = [Self.plaintextModulus]
        let scalingFactor = ClientConfig<Scheme>.maxScalingFactor(
            distanceMetric: .dotProduct,
            vectorDimension: vectorDimension,
            plaintextModuli: plaintextModuli)

        let evaluationKeyConfig = try MatrixMultiplication.evaluationKeyConfig(
            plaintextMatrixDimensions: MatrixDimensions(
                rowCount: rowCount, columnCount: vectorDimension),
            maxQueryCount: 1,
            encryptionParameters: encryptionParameters,
            scheme: Scheme.self)

        let clientConfig = try ClientConfig<Scheme>(
            encryptionParameters: encryptionParameters,
            scalingFactor: scalingFactor,
            queryPacking: .denseRow,
            vectorDimension: vectorDimension,
            evaluationKeyConfig: evaluationKeyConfig,
            distanceMetric: .dotProduct)
        let serverConfig = ServerConfig(
            clientConfig: clientConfig,
            databasePacking: .diagonal(
                babyStepGiantStep: BabyStepGiantStep(vectorDimension: vectorDimension)))
        let context = try Scheme.Context(encryptionParameters: encryptionParameters)

        // Create test database
        let rawVectors: [[Float]] = (0..<rowCount).map { rowIndex in
            (0..<vectorDimension).map { colIndex in
                let norm = (0..<vectorDimension).map { c in
                    Float(c + rowIndex) * Float(c + rowIndex)
                }.reduce(0, +).squareRoot()
                return Float(colIndex + rowIndex) * (rowIndex.isMultiple(of: 2) ? 1 : -1) / max(norm, 1)
            }
        }
        let database = Database(rows: rawVectors.enumerated().map { i, vec in
            DatabaseRow(entryId: UInt64(i), entryMetadata: [], vector: vec)
        })

        // Step 1: Quantize
        let quantized = QuantizedDatabase<Scheme>(database: database, config: serverConfig)

        // Step 2: Serialize (this is what goes into ORAM)
        let bytes = quantized.serializeVectors()

        // Step 3: Deserialize (proxy reads from ORAM)
        let restored = QuantizedDatabase<Scheme>.deserializeVectors(
            from: bytes,
            rowCount: quantized.rowCount,
            vectorDimension: quantized.vectorDimension,
            entryIds: quantized.entryIds,
            entryMetadatas: quantized.entryMetadatas)

        // Verify round-trip
        #expect(restored.signedValues == quantized.signedValues)
        #expect(restored.entryIds == quantized.entryIds)

        // Step 4: Fast encode
        let encoder = try FastPlaintextEncoder<Scheme>(
            config: serverConfig, context: context, rowCount: rowCount)
        let processed = try encoder.encodeDatabase(restored, context: context)

        // Step 5: Query and verify
        let client = try Client<Scheme>(config: clientConfig, contexts: [context])
        let secretKey = try client.generateSecretKey()
        let evalKey = try client.generateEvaluationKey(using: secretKey)
        let queryVectors = Array2d(data: [rawVectors[0]])
        let query = try client.generateQuery(for: queryVectors, using: secretKey)

        let server = try Server(database: processed)
        let response = try await server.computeResponse(to: query, using: evalKey)
        let distances = try client.decrypt(response: response, using: secretKey)

        // Verify against standard pipeline
        let standardProcessed = try await database.process(config: serverConfig, contexts: [context])
        let standardServer = try Server(database: standardProcessed)
        let standardResponse = try await standardServer.computeResponse(to: query, using: evalKey)
        let standardDistances = try client.decrypt(response: standardResponse, using: secretKey)

        var maxDiff: Float = 0
        for i in 0..<distances.distances.data.count {
            let diff = abs(distances.distances.data[i] - standardDistances.distances.data[i])
            maxDiff = max(maxDiff, diff)
        }
        #expect(maxDiff == 0, Comment(rawValue: "Round-trip differs: maxDiff=\(maxDiff)"))
    }
}
