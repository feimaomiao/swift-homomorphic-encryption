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

public import AsyncAlgorithms
public import HomomorphicEncryption
public import PrivateNearestNeighborSearch

/// A database of diagonal/BSGS-packed ciphertext matrices, encrypted under one secret key.
///
/// Ciphertext-database analogue of ``ProcessedDatabase``. An offshore server holds one of these
/// plus the matching evaluation key; it can compute encrypted similarity scores against
/// encrypted queries without ever holding the secret key.
public struct EncryptedProcessedDatabase<Scheme: HeScheme>: Sendable {
    /// Contexts, one per plaintext CRT modulus. Match the plaintext-modulus layout in
    /// ``ServerConfig/encryptionParameters``.
    public let contexts: [Scheme.Context]

    /// Encrypted database matrices, one per plaintext CRT modulus. Each matrix holds the
    /// diagonal/BSGS-packed database rows under the client's secret key.
    public let encryptedMatrices: [CiphertextMatrix<Scheme, Scheme.CanonicalCiphertextFormat>]

    /// Unique identifier for each database entry. Rides alongside the response so the client
    /// can pair `(entryId, score)` tuples without re-learning the ordering.
    public let entryIds: [UInt64]

    /// Metadata for each database entry. Same pairing guarantee as `entryIds`.
    public let entryMetadatas: [[UInt8]]

    /// Server configuration (distance metric, scaling factor, packings, evaluation-key config).
    public let serverConfig: ServerConfig<Scheme>

    /// Creates a new ``EncryptedProcessedDatabase``.
    /// - Throws: Error if `contexts` don't match the server config's encryption parameters.
    @inlinable
    public init(
        contexts: [Scheme.Context],
        encryptedMatrices: [CiphertextMatrix<Scheme, Scheme.CanonicalCiphertextFormat>],
        entryIds: [UInt64],
        entryMetadatas: [[UInt8]],
        serverConfig: ServerConfig<Scheme>) throws
    {
        try serverConfig.validateContexts(contexts: contexts)
        self.contexts = contexts
        self.encryptedMatrices = encryptedMatrices
        self.entryIds = entryIds
        self.entryMetadatas = entryMetadatas
        self.serverConfig = serverConfig
    }
}

/// Ciphertext × ciphertext private nearest neighbor server.
///
/// Mirrors ``PrivateNearestNeighborSearch/Server`` but consumes an
/// ``EncryptedProcessedDatabase``. The offshore server never holds the secret key — only the
/// encrypted database, the evaluation key (shipped alongside), and the encrypted query.
public struct Server<Scheme: HeScheme>: Sendable {
    /// The encrypted database.
    public let database: EncryptedProcessedDatabase<Scheme>

    /// Server configuration.
    public var config: ServerConfig<Scheme> {
        database.serverConfig
    }

    /// Client-facing configuration.
    public var clientConfig: ClientConfig<Scheme> {
        config.clientConfig
    }

    /// Evaluation-key configuration required for the ciphertext × ciphertext kernel.
    public var evaluationKeyConfig: EvaluationKeyConfig {
        clientConfig.evaluationKeyConfig
    }

    /// One context per plaintext modulus.
    public var contexts: [Scheme.Context] {
        database.contexts
    }

    /// Creates a new ``Server``.
    /// - Parameter database: Encrypted processed database.
    /// - Throws: Error if construction fails.
    @inlinable
    public init(database: EncryptedProcessedDatabase<Scheme>) throws {
        self.database = database
    }

    /// Computes the encrypted response to an encrypted query.
    ///
    /// Mirrors ``PrivateNearestNeighborSearch/Server/computeResponse(to:using:)`` but uses
    /// the ciphertext × ciphertext kernel. After the kernel, `modSwitchDownToSingle` reduces
    /// the response ciphertexts to a single coefficient modulus for much smaller wire
    /// transfer (~4× smaller for a 4-modulus parameter set) at the cost of some noise budget.
    /// - Parameters:
    ///   - query: The encrypted query (one ciphertext matrix per plaintext modulus).
    ///   - evaluationKey: The evaluation key shipped by the client. Must contain the BSGS
    ///     Galois elements and a relinearization key.
    /// - Returns: The encrypted similarity scores (one ciphertext matrix per plaintext modulus)
    ///   paired with the database entry IDs and metadata.
    /// - Throws: Error upon dimension / context / packing mismatch, missing keys, or kernel
    ///   failure.
    @inlinable
    public func computeResponse(
        to query: Query<Scheme>,
        using evaluationKey: EvaluationKey<Scheme>) async throws -> Response<Scheme>
    {
        guard query.ciphertextMatrices.count == database.encryptedMatrices.count else {
            throw PnnsError.invalidQuery(reason: InvalidQueryReason.wrongCiphertextMatrixCount(
                got: query.ciphertextMatrices.count,
                expected: database.encryptedMatrices.count))
        }
        let canonicalQueries: [CiphertextMatrix<Scheme, Scheme.CanonicalCiphertextFormat>] =
            try await .init(query.ciphertextMatrices.async.map { ciphertextMatrix in
                try ciphertextMatrix.convertToCanonicalFormat()
            })
        let responseMatrices: [CiphertextMatrix<Scheme, Coeff>] = try await .init(
            zip(canonicalQueries, database.encryptedMatrices).async.map { queryMatrix, dbMatrix in
                var responseMatrix = try await dbMatrix.mulTranspose(
                    matrix: queryMatrix, using: evaluationKey)
                // Shrink the response before shipping. Mirrors Server.computeResponse's
                // post-kernel modulus reduction at Server.swift:77.
                try await responseMatrix.modSwitchDownToSingle()
                return try await responseMatrix.convertToCoeffFormat()
            })
        return Response(
            ciphertextMatrices: responseMatrices,
            entryIds: database.entryIds,
            entryMetadatas: database.entryMetadatas)
    }
}

extension Database {
    /// Processes and encrypts the database for the ciphertext × ciphertext path.
    ///
    /// Parallel to ``Database/process(config:contexts:)``, but produces an encrypted database
    /// instead of a plaintext one. The caller must hold the secret key.
    /// - Parameters:
    ///   - config: Server configuration.
    ///   - contexts: Contexts, one per plaintext modulus. If empty, contexts are derived from
    ///     `config.encryptionParameters`.
    ///   - secretKey: Secret key to encrypt under. Must match `contexts[0]`.
    /// - Returns: The encrypted processed database.
    /// - Throws: Error upon validation or encryption failure.
    @inlinable
    public func processAndEncrypt<Scheme: HeScheme>(
        config: ServerConfig<Scheme>,
        secretKey: SecretKey<Scheme>,
        contexts providedContexts: [Scheme.Context] = []) throws -> EncryptedProcessedDatabase<Scheme>
    {
        var contexts = providedContexts
        if contexts.isEmpty {
            contexts = try config.encryptionParameters.map { params in
                try Scheme.Context(encryptionParameters: params)
            }
        }
        try config.validateContexts(contexts: contexts)

        let vectors = Array2d(data: rows.map(\.vector))
        let roundedVectors: Array2d<Scheme.SignedScalar> = switch config.distanceMetric {
        case .cosineSimilarity:
            vectors.normalizedScaledAndRounded(scalingFactor: Float(config.scalingFactor))
        case .dotProduct:
            vectors.scaled(by: Float(config.scalingFactor)).rounded()
        }

        let shouldReduce = contexts.count > 1
        let encryptedMatrices: [CiphertextMatrix<Scheme, Scheme.CanonicalCiphertextFormat>] =
            try contexts.map { context in
                try PlaintextMatrix<Scheme, Coeff>(
                    context: context,
                    dimensions: MatrixDimensions(roundedVectors.shape),
                    packing: config.databasePacking,
                    signedValues: roundedVectors.data,
                    reduce: shouldReduce).encrypt(using: secretKey)
            }
        let hasMetadata = rows.contains { !$0.entryMetadata.isEmpty }
        return try EncryptedProcessedDatabase(
            contexts: contexts,
            encryptedMatrices: encryptedMatrices,
            entryIds: rows.map(\.entryId),
            entryMetadatas: hasMetadata ? rows.map(\.entryMetadata) : [],
            serverConfig: config)
    }
}
