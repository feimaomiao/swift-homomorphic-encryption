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
public import PrivateNearestNeighborSearch

/// A serialized ``EncryptedProcessedDatabase``.
///
/// Single-file counterpart of the multi-file layout used by
/// `CTPNNSProcessDatabase`, mirroring ``SerializedProcessedDatabase``.
public struct SerializedEncryptedProcessedDatabase<Scheme: HeScheme>: Sendable {
    /// The encrypted vectors in the database, one per plaintext CRT modulus.
    public let encryptedMatrices: [SerializedCiphertextMatrix<Scheme.Scalar>]

    /// Unique identifier for each database entry.
    public let entryIds: [UInt64]

    /// Associated metadata for each database entry.
    public let entryMetadatas: [[UInt8]]

    /// Server configuration.
    public let serverConfig: ServerConfig<Scheme>

    /// Creates a new ``SerializedEncryptedProcessedDatabase``.
    public init(
        encryptedMatrices: [SerializedCiphertextMatrix<Scheme.Scalar>],
        entryIds: [UInt64],
        entryMetadatas: [[UInt8]],
        serverConfig: ServerConfig<Scheme>)
    {
        self.encryptedMatrices = encryptedMatrices
        self.entryIds = entryIds
        self.entryMetadatas = entryMetadatas
        self.serverConfig = serverConfig
    }
}

extension EncryptedProcessedDatabase {
    /// Initializes an ``EncryptedProcessedDatabase`` from a
    /// ``SerializedEncryptedProcessedDatabase``.
    /// - Parameters:
    ///   - serialized: Serialized encrypted processed database.
    ///   - contexts: Contexts for HE computation, one per plaintext modulus.
    ///     Pass empty to derive contexts from the server config.
    /// - Throws: Error upon failure to load the database.
    public init(
        from serialized: SerializedEncryptedProcessedDatabase<Scheme>,
        contexts: [Scheme.Context] = []) throws
    {
        var contexts = contexts
        if contexts.isEmpty {
            contexts = try serialized.serverConfig.encryptionParameters.map { encryptionParameters in
                try Scheme.Context(encryptionParameters: encryptionParameters)
            }
        }
        try serialized.serverConfig.validateContexts(contexts: contexts)

        let encryptedMatrices = try zip(serialized.encryptedMatrices, contexts)
            .map { matrix, context in
                try CiphertextMatrix<Scheme, Scheme.CanonicalCiphertextFormat>(
                    deserialize: matrix, context: context)
            }
        try self.init(
            contexts: contexts,
            encryptedMatrices: encryptedMatrices,
            entryIds: serialized.entryIds,
            entryMetadatas: serialized.entryMetadatas,
            serverConfig: serialized.serverConfig)
    }

    /// Serializes the encrypted processed database.
    /// - Returns: The serialized encrypted processed database.
    /// - Throws: Error upon failure to serialize.
    public func serialize() throws -> SerializedEncryptedProcessedDatabase<Scheme> {
        try SerializedEncryptedProcessedDatabase(
            encryptedMatrices: encryptedMatrices.map { matrix in try matrix.serialize() },
            entryIds: entryIds,
            entryMetadatas: entryMetadatas,
            serverConfig: serverConfig)
    }
}
