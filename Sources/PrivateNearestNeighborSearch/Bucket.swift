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

/// A database labeled with a cluster identifier.
public struct ClusterDatabase: Codable, Equatable, Hashable, Sendable {
    /// Cluster identifier.
    public let clusterId: UInt32
    /// The database for this cluster.
    public let database: Database

    /// Creates a new ``ClusterDatabase``.
    /// - Parameters:
    ///   - clusterId: Cluster identifier.
    ///   - database: The database for this cluster.
    public init(clusterId: UInt32, database: Database) {
        self.clusterId = clusterId
        self.database = database
    }
}

/// A collection of cluster databases for oblivious nearest neighbor search.
public struct Bucket: Codable, Equatable, Hashable, Sendable {
    /// The cluster databases in this bucket.
    public let clusters: [ClusterDatabase]

    /// Creates a new ``Bucket``.
    /// - Parameter clusters: The cluster databases.
    public init(clusters: [ClusterDatabase]) {
        self.clusters = clusters
    }
}

/// A processed cluster database ready for encrypted evaluation.
public struct ProcessedClusterDatabase<Scheme: HeScheme>: Sendable {
    /// Cluster identifier.
    public let clusterId: UInt32
    /// The processed database for this cluster.
    public let database: ProcessedDatabase<Scheme>

    /// Creates a new ``ProcessedClusterDatabase``.
    /// - Parameters:
    ///   - clusterId: Cluster identifier.
    ///   - database: The processed database for this cluster.
    public init(clusterId: UInt32, database: ProcessedDatabase<Scheme>) {
        self.clusterId = clusterId
        self.database = database
    }
}

/// A processed bucket ready for encrypted evaluation.
public struct ProcessedBucket<Scheme: HeScheme>: Sendable {
    /// The processed cluster databases.
    public let clusters: [ProcessedClusterDatabase<Scheme>]
    /// Server configuration shared by all clusters.
    public let serverConfig: ServerConfig<Scheme>

    /// Creates a new ``ProcessedBucket``.
    /// - Parameters:
    ///   - clusters: The processed cluster databases.
    ///   - serverConfig: Server configuration shared by all clusters.
    public init(clusters: [ProcessedClusterDatabase<Scheme>], serverConfig: ServerConfig<Scheme>) {
        self.clusters = clusters
        self.serverConfig = serverConfig
    }
}

/// An encrypted response tagged with the cluster it came from.
public struct ClusterResponse<Scheme: HeScheme>: Sendable {
    /// Cluster identifier.
    public let clusterId: UInt32
    /// The encrypted response for this cluster.
    public let response: Response<Scheme>

    /// Creates a new ``ClusterResponse``.
    /// - Parameters:
    ///   - clusterId: Cluster identifier.
    ///   - response: The encrypted response for this cluster.
    public init(clusterId: UInt32, response: Response<Scheme>) {
        self.clusterId = clusterId
        self.response = response
    }
}

extension Bucket {
    /// Processes all databases in the bucket for nearest neighbor search.
    /// - Parameters:
    ///   - config: Server configuration shared by all clusters.
    ///   - contexts: Contexts for HE computation, one per plaintext modulus.
    /// - Returns: The processed bucket.
    /// - Throws: Error upon failure to process any database.
    @inlinable
    public func process<Scheme: HeScheme>(
        config: ServerConfig<Scheme>,
        contexts: [Scheme.Context] = []) async throws -> ProcessedBucket<Scheme>
    {
        var contexts = contexts
        if contexts.isEmpty {
            contexts = try config.encryptionParameters.map { encryptionParameters in
                try Scheme.Context(encryptionParameters: encryptionParameters)
            }
        }
        let resolvedContexts = contexts
        let nonEmptyClusters = clusters.filter { !$0.database.rows.isEmpty }
        let processedClusters: [ProcessedClusterDatabase<Scheme>] = try await parallelMap(
            count: nonEmptyClusters.count)
        { index in
            let cluster = nonEmptyClusters[index]
            let processed = try await cluster.database.process(config: config, contexts: resolvedContexts)
            return ProcessedClusterDatabase(clusterId: cluster.clusterId, database: processed)
        }
        return ProcessedBucket(clusters: processedClusters, serverConfig: config)
    }
}

/// Server that evaluates encrypted queries against all clusters in a bucket.
public struct BucketServer<Scheme: HeScheme>: Sendable {
    /// The processed bucket.
    public let bucket: ProcessedBucket<Scheme>

    /// Creates a new ``BucketServer``.
    /// - Parameter bucket: The processed bucket.
    public init(bucket: ProcessedBucket<Scheme>) {
        self.bucket = bucket
    }

    /// Evaluates a query against every cluster in the bucket.
    /// - Parameters:
    ///   - query: The encrypted query.
    ///   - evaluationKey: Evaluation key for HE computation.
    /// - Returns: One ``ClusterResponse`` per cluster.
    /// - Throws: Error upon failure to compute any response.
    @inlinable
    public func computeResponses(
        to query: Query<Scheme>,
        using evaluationKey: EvaluationKey<Scheme>) async throws -> [ClusterResponse<Scheme>]
    {
        try await parallelMap(count: bucket.clusters.count) { index in
            let cluster = bucket.clusters[index]
            let server = try Server(database: cluster.database)
            let response = try await server.computeResponse(to: query, using: evaluationKey)
            return ClusterResponse(clusterId: cluster.clusterId, response: response)
        }
    }
}
