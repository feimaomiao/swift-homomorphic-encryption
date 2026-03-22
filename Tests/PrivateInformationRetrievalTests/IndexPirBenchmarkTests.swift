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

import HomomorphicEncryption
import PrivateInformationRetrieval
import Testing

struct IndexPirBenchmarkTests {
    @Test
    func pirScaling() async throws {
        let rlweParams: [PredefinedRlweParameters] = [
            .n_4096_logq_27_28_28_logt_5,
        ]

        struct BenchConfig {
            let entryCount: Int
            let entrySize: Int
        }
        let configs: [BenchConfig] = [
            BenchConfig(entryCount: 3000, entrySize: 500),
            BenchConfig(entryCount: 10000, entrySize: 500),
            BenchConfig(entryCount: 3000, entrySize: 2000),
            BenchConfig(entryCount: 10000, entrySize: 2000),
            BenchConfig(entryCount: 1000, entrySize: 10000),
            BenchConfig(entryCount: 3000, entrySize: 10000),
            BenchConfig(entryCount: 10000, entrySize: 10000),
        ]

        for params in rlweParams {
            let encryptionParameters = try EncryptionParameters<UInt64>(from: params)
            let context = try Bfv<UInt64>.Context(encryptionParameters: encryptionParameters)

            for config in configs {
                let pirConfig = try IndexPirConfig(
                    entryCount: config.entryCount,
                    entrySizeInBytes: config.entrySize,
                    dimensionCount: 2,
                    batchSize: 1,
                    unevenDimensions: true,
                    keyCompression: .noCompression,
                    encodingEntrySize: false)

                let parameter = MulPir<Bfv<UInt64>>.generateParameter(
                    config: pirConfig, with: context)

                let database: [[UInt8]] = (0..<config.entryCount).map { _ in
                    (0..<config.entrySize).map { _ in UInt8.random(in: 0...255) }
                }

                let processedDb = try await MulPirServer<PirUtil<Bfv<UInt64>>>.process(
                    database: database, with: context, using: parameter)

                let server = try MulPirServer<PirUtil<Bfv<UInt64>>>(
                    parameter: parameter, context: context, database: processedDb)
                let client = MulPirClient<PirUtil<Bfv<UInt64>>>(
                    parameter: parameter, context: context)

                let secretKey = try context.generateSecretKey()
                let evalKey = try client.generateEvaluationKey(using: secretKey)
                let queryIndex = Int.random(in: 0..<config.entryCount)

                let query = try client.generateQuery(at: [queryIndex], using: secretKey)
                let response = try await server.computeResponse(to: query, using: evalKey)
                let decrypted = try client.decrypt(
                    response: response, at: [queryIndex], using: secretKey)
                #expect(decrypted[0] == database[queryIndex])
            }
        }
    }
}
