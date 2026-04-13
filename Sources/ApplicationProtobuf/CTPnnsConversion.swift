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

public import CTPrivateNearestNeighbourSearch
public import HomomorphicEncryption
public import PrivateNearestNeighborSearch
import Foundation

extension Apple_SwiftHomomorphicEncryption_Pnns_V1_SerializedEncryptedProcessedDatabase {
    /// Converts the protobuf object to a native type.
    /// - Returns: The converted native type.
    /// - Throws: Error upon upon invalid object.
    public func native<Scheme: HeScheme>() throws -> SerializedEncryptedProcessedDatabase<Scheme> {
        try SerializedEncryptedProcessedDatabase(
            encryptedMatrices: encryptedMatrices.map { matrix in try matrix.native() },
            entryIds: entryIds,
            entryMetadatas: entryMetadatas.map { metadata in Array(metadata) },
            serverConfig: serverConfig.native())
    }
}

extension SerializedEncryptedProcessedDatabase {
    /// Converts the native object into a protobuf object.
    /// - Returns: The converted protobuf object.
    /// - Throws: Error upon unsupported object.
    public func proto() throws -> Apple_SwiftHomomorphicEncryption_Pnns_V1_SerializedEncryptedProcessedDatabase {
        try Apple_SwiftHomomorphicEncryption_Pnns_V1_SerializedEncryptedProcessedDatabase
            .with { protoDatabase in
                protoDatabase.encryptedMatrices = try encryptedMatrices.map { matrix in try matrix.proto() }
                protoDatabase.entryIds = entryIds
                protoDatabase.entryMetadatas = entryMetadatas.map { metadata in Data(metadata) }
                protoDatabase.serverConfig = try serverConfig.proto()
            }
    }
}
