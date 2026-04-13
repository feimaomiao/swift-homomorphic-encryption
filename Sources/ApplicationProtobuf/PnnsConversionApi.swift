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
import Foundation
import HomomorphicEncryptionProtobuf

extension Apple_SwiftHomomorphicEncryption_Api_Pnns_V1_PNNSShardResponse {
    /// Converts the protobuf object to a native type.
    /// - Parameters:
    ///   - contexts: Contexts to associate with the native type; one context per plaintext modulus.
    ///   - moduliCount: Number of coefficient moduli in each serialized ciphertext. Defaults to 1
    ///     (single-modulus responses from `computeResponse(modSwitchDown: true)`). Pass `nil` to
    ///     use all moduli from the context, which is needed when responses were computed with
    ///     `modSwitchDown: false` to preserve noise budget for aggregation.
    public func native<Scheme: HeScheme>(
        contexts: [Scheme.Context],
        moduliCount: Int? = 1) throws -> Response<Scheme>
    {
        precondition(contexts.count == reply.count)
        let matrices: [CiphertextMatrix<Scheme, Coeff>] = try zip(reply, contexts).map { matrix, context in
            let serialized: SerializedCiphertextMatrix<Scheme.Scalar> = try matrix.native()
            return try CiphertextMatrix(deserialize: serialized, context: context, moduliCount: moduliCount)
        }
        return Response(
            ciphertextMatrices: matrices,
            entryIds: entryIds,
            entryMetadatas: entryMetadatas.map { metadata in Array(metadata) })
    }
}

extension Response {
    /// Converts the native object into a protobuf object.
    /// - Returns: The converted protobuf object.
    /// - Throws: Error upon unsupported object.
    public func proto() throws -> Apple_SwiftHomomorphicEncryption_Api_Pnns_V1_PNNSShardResponse {
        try Apple_SwiftHomomorphicEncryption_Api_Pnns_V1_PNNSShardResponse.with { pnnsResponse in
            pnnsResponse.reply = try ciphertextMatrices.map { matrix in
                try matrix.serialize(forDecryption: true).proto()
            }
            pnnsResponse.entryIds = entryIds
            pnnsResponse.entryMetadatas = entryMetadatas.map { bytes in Data(bytes) }
        }
    }
}
