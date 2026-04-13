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

public import HomomorphicEncryption

/// Private nearest neighbor server.
public struct Server<Scheme: HeScheme>: Sendable {
    /// The database.
    public let database: ProcessedDatabase<Scheme>

    /// Configuration.
    public var config: ServerConfig<Scheme> {
        database.serverConfig
    }

    /// Client configuration.
    public var clientConfig: ClientConfig<Scheme> {
        config.clientConfig
    }

    /// Configuration needed for private nearest neighbor search.
    public var evaluationKeyConfig: EvaluationKeyConfig {
        config.clientConfig.evaluationKeyConfig
    }

    /// One context per plaintext modulus.
    public var contexts: [Scheme.Context] {
        database.contexts
    }

    /// Creates a new ``Server``.
    /// - Parameter database: Processed database.
    /// - Throws: Error upon failure to create the server.
    @inlinable
    public init(database: ProcessedDatabase<Scheme>) throws {
        self.database = database
    }

    /// Compute the encrypted response to a query.
    /// - Parameters:
    ///   - query: Query.
    ///   - evaluationKey: Evaluation key to aid in the server computation.
    ///   - modSwitchDown: If `true` (the default), mod-switches the response ciphertexts down to a
    ///     single coefficient modulus, minimizing response size for immediate decryption.
    ///
    ///     Set to `false` when the response will be aggregated (ciphertext-added) with other
    ///     responses before decryption. Keeping ciphertexts at a higher modulus level preserves
    ///     noise budget for the subsequent additions.
    ///
    ///     **Why this matters:** For BFV with small coefficient moduli (e.g. `n_4096_logq_27_28_28_logt_17`),
    ///     the lowest modulus q₀ provides only ~10 bits of headroom above the plaintext modulus.
    ///     Adding 4 ciphertexts at this level can overflow the noise budget, producing incorrect
    ///     decryption. Deferring the mod-switch until after aggregation keeps ciphertexts at a
    ///     level with ~38+ bits of headroom, which is sufficient for many additions.
    /// - Returns: The response.
    /// - Throws: Error upon failure to compute a response.
    @inlinable
    public func computeResponse(to query: Query<Scheme>,
                                using evaluationKey: EvaluationKey<Scheme>,
                                modSwitchDown: Bool = true) async throws -> Response<Scheme>
    {
        guard query.ciphertextMatrices.count == database.plaintextMatrices.count else {
            throw PnnsError.invalidQuery(reason: InvalidQueryReason.wrongCiphertextMatrixCount(
                got: query.ciphertextMatrices.count,
                expected: database.plaintextMatrices.count))
        }
        let ciphertextMatrixCount = query.ciphertextMatrices.count
        let responseMatrices: [CiphertextMatrix<Scheme, Coeff>] = try await parallelMap(
            count: ciphertextMatrixCount)
        { index in
            let ciphertextMatrix = try await query.ciphertextMatrices[index].convertToCanonicalFormat()
            let plaintextMatrix = database.plaintextMatrices[index]
            var responseMatrix = try await plaintextMatrix.mulTranspose(
                matrix: ciphertextMatrix,
                using: evaluationKey)
            if modSwitchDown {
                // Reduce response size by mod-switching to a single modulus.
                // Only safe when the response will be decrypted directly (no further additions).
                try await responseMatrix.modSwitchDownToSingle()
            }
            return try await responseMatrix.convertToCoeffFormat()
        }

        return Response(
            ciphertextMatrices: responseMatrices,
            entryIds: database.entryIds,
            entryMetadatas: database.entryMetadatas)
    }
}
