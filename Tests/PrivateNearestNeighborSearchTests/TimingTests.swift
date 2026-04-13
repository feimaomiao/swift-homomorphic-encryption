// Copyright 2024-2026 Apple Inc. and the Swift Homomorphic Encryption project authors
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

import HomomorphicEncryption
import PrivateNearestNeighborSearch
import Testing

/// Timing tests for PNNS computeResponse wall time across a matrix of configurations.
///
/// Run with: swift test --filter TimingTests -c release
struct TimingTests {
    typealias Scheme = Bfv<UInt64>
    typealias Scalar = UInt64

    struct Config: CustomStringConvertible {
        let rowCount: Int
        let vectorDimension: Int
        let queryCount: Int

        var description: String {
            "batch=\(queryCount), rows=\(rowCount), dim=\(vectorDimension)"
        }
    }

    struct Setup {
        let server: Server<Scheme>
        let client: Client<Scheme>
        let query: Query<Scheme>
        let evaluationKey: EvaluationKey<Scheme>
        let secretKey: SecretKey<Scheme>
    }

    static let configs: [Config] = {
        var result: [Config] = []
        for rows in [1024, 4096] {
            for dim in [32, 64, 128, 256, 512] {
                for batch in [1, 4, 16] {
                    result.append(Config(rowCount: rows, vectorDimension: dim, queryCount: batch))
                }
            }
        }
        return result
    }()

    static let trials = 100

    /// Helper to build a PNNS setup with realistic parameters.
    static func buildSetup(config: Config) async throws -> Setup {
        let degree = 4096
        let plaintextModuli = try Scalar.generatePrimes(
            significantBitCounts: [17],
            preferringSmall: true,
            nttDegree: degree)
        let coefficientModuli = try Scalar.generatePrimes(
            significantBitCounts: [27, 28, 28],
            preferringSmall: false,
            nttDegree: degree)
        let encryptionParameters = try EncryptionParameters<Scalar>(
            polyDegree: degree,
            plaintextModulus: plaintextModuli[0],
            coefficientModuli: coefficientModuli,
            errorStdDev: .stdDev32,
            securityLevel: .unchecked)

        let dimensions = try MatrixDimensions(rowCount: config.rowCount, columnCount: config.vectorDimension)
        let evaluationKeyConfig = try MatrixMultiplication.evaluationKeyConfig(
            plaintextMatrixDimensions: dimensions,
            maxQueryCount: config.queryCount,
            encryptionParameters: encryptionParameters,
            scheme: Scheme.self)
        let scalingFactor = ClientConfig<Scheme>.maxScalingFactor(
            distanceMetric: .cosineSimilarity,
            vectorDimension: config.vectorDimension,
            plaintextModuli: plaintextModuli)
        let clientConfig = try ClientConfig<Scheme>(
            encryptionParameters: encryptionParameters,
            scalingFactor: scalingFactor,
            queryPacking: .denseRow,
            vectorDimension: config.vectorDimension,
            evaluationKeyConfig: evaluationKeyConfig,
            distanceMetric: .cosineSimilarity)
        let babyStepGiantStep = BabyStepGiantStep(vectorDimension: config.vectorDimension)
        let serverConfig = ServerConfig(
            clientConfig: clientConfig,
            databasePacking: .diagonal(babyStepGiantStep: babyStepGiantStep))

        let rows = (0..<config.rowCount).map { i in
            DatabaseRow(
                entryId: UInt64(i),
                entryMetadata: [],
                vector: (0..<config.vectorDimension).map { Float($0 + i) * (i.isMultiple(of: 2) ? 1 : -1) })
        }
        let database = Database(rows: rows)
        let processed = try await database.process(config: serverConfig)
        let server = try Server(database: processed)
        let client = try Client(config: clientConfig, contexts: processed.contexts)

        let secretKey = try client.generateSecretKey()
        let evaluationKey = try client.generateEvaluationKey(using: secretKey)
        let queryVectors = Array2d(data: database.rows.prefix(config.queryCount).map { row in row.vector })
        let query = try client.generateQuery(for: queryVectors, using: secretKey)

        return Setup(
            server: server,
            client: client,
            query: query,
            evaluationKey: evaluationKey,
            secretKey: secretKey)
    }

    @Test
    func timingMatrix() async throws {
        print("\n>>> TIMING MATRIX START (trials=\(Self.trials))")
        print(">>> \(String(repeating: "=", count: 80))")

        for config in Self.configs {
            let setup = try await Self.buildSetup(config: config)
            let server = setup.server
            let client = setup.client
            let query = setup.query
            let evaluationKey = setup.evaluationKey
            let secretKey = setup.secretKey

            // Warmup (3 iterations)
            for _ in 0..<3 {
                let warmup = try await server.computeResponse(to: query, using: evaluationKey)
                _ = try client.decrypt(response: warmup, using: secretKey)
            }

            // Timed runs
            let clock = ContinuousClock()
            var durations: [Double] = []
            durations.reserveCapacity(Self.trials)

            for _ in 0..<Self.trials {
                let elapsed = try await clock.measure {
                    let response = try await server.computeResponse(to: query, using: evaluationKey)
                    _ = try client.decrypt(response: response, using: secretKey)
                }
                let ms = Double(elapsed.components.attoseconds) / 1e15
                durations.append(ms)
            }

            durations.sort()
            let avg = durations.reduce(0, +) / Double(Self.trials)
            let median = durations[Self.trials / 2]
            let p5 = durations[Self.trials * 5 / 100]
            let p95 = durations[Self.trials * 95 / 100]
            let minValue = durations.min() ?? 0
            let maxValue = durations.max() ?? 0

            print(
                ">>> [\(config)] avg=\(f(avg)) median=\(f(median)) p5=\(f(p5)) p95=\(f(p95)) " +
                    "min=\(f(minValue)) max=\(f(maxValue))")
        }

        print(">>> \(String(repeating: "=", count: 80))")
        print(">>> TIMING MATRIX END")
    }

    private func f(_ value: Double) -> String {
        String(format: "%.1f", value)
    }
}
