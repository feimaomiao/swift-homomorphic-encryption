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

import Foundation
import HomomorphicEncryption
@testable import PrivateNearestNeighborSearch
import Testing

struct BucketTests {
    private typealias Scheme = Bfv<UInt64>

    private static let polyDegree = 8192
    private static let plaintextModulus: UInt64 = 536_903_681
    private static let extraPlaintextModulus: UInt64 = 65537
    private static let coefficientModuli: [UInt64] = [
        36_028_797_018_652_673,
        36_028_797_017_571_329,
        36_028_797_017_456_641,
    ]
    private static let vectorDimension = 128
    private static let rowCount = 64

    private static func makeEncryptionParameters() throws -> (
        params: EncryptionParameters<UInt64>,
        plaintextModuli: [UInt64])
    {
        let params = try EncryptionParameters<UInt64>(
            polyDegree: polyDegree,
            plaintextModulus: plaintextModulus,
            coefficientModuli: coefficientModuli,
            errorStdDev: .stdDev32,
            securityLevel: .unchecked)
        return (params, [plaintextModulus, extraPlaintextModulus])
    }

    private static func makeConfig() throws -> (
        clientConfig: ClientConfig<Scheme>,
        serverConfig: ServerConfig<Scheme>)
    {
        let (encryptionParameters, plaintextModuli) = try makeEncryptionParameters()
        let scalingFactor = ClientConfig<Scheme>.maxScalingFactor(
            distanceMetric: .cosineSimilarity,
            vectorDimension: vectorDimension,
            plaintextModuli: plaintextModuli)
        let evaluationKeyConfig = try MatrixMultiplication.evaluationKeyConfig(
            plaintextMatrixDimensions: MatrixDimensions(rowCount: rowCount, columnCount: vectorDimension),
            maxQueryCount: 1,
            encryptionParameters: encryptionParameters,
            scheme: Scheme.self)
        let clientConfig = try ClientConfig<Scheme>(
            encryptionParameters: encryptionParameters,
            scalingFactor: scalingFactor,
            queryPacking: .denseRow,
            vectorDimension: vectorDimension,
            evaluationKeyConfig: evaluationKeyConfig,
            distanceMetric: .cosineSimilarity,
            extraPlaintextModuli: Array(plaintextModuli[1...]))
        let serverConfig = ServerConfig(
            clientConfig: clientConfig,
            databasePacking: .diagonal(
                babyStepGiantStep: BabyStepGiantStep(vectorDimension: vectorDimension)))
        return (clientConfig, serverConfig)
    }

    private static func syntheticRows(count: Int, startId: UInt64 = 0) -> [DatabaseRow] {
        (0..<count).map { i in
            let vector = (0..<vectorDimension).map { col in
                Float(col + i) * (i.isMultiple(of: 2) ? 1 : -1)
            }
            return DatabaseRow(entryId: startId + UInt64(i), entryMetadata: [], vector: vector)
        }
    }

    private static func dot(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).map(*).reduce(0, +)
    }

    private static func normalizeRows(_ vectors: [[Float]]) -> [[Float]] {
        vectors.map { row in
            let norm = row.map { $0 * $0 }.reduce(0, +).squareRoot()
            guard norm > 0 else {
                return row
            }
            return row.map { $0 / norm }
        }
    }

    @Test
    func singleClusterMatchesServer() async throws {
        let (_, serverConfig) = try Self.makeConfig()
        let rows = Self.syntheticRows(count: Self.rowCount)
        let database = Database(rows: rows)

        // Direct server path
        let processedDB = try await database.process(config: serverConfig)
        let server = try Server(database: processedDB)
        let client = try Client(config: serverConfig.clientConfig, contexts: processedDB.contexts)
        let secretKey = try client.generateSecretKey()
        let evalKey = try client.generateEvaluationKey(using: secretKey)
        let queryVectors = Array2d(data: [rows[0].vector])
        let query = try client.generateQuery(for: queryVectors, using: secretKey)
        let directResponse = try await server.computeResponse(to: query, using: evalKey)
        let directDistances = try client.decrypt(response: directResponse, using: secretKey)

        // Bucket path with one cluster
        let bucket = Bucket(clusters: [ClusterDatabase(clusterId: 0, database: database)])
        let processedBucket = try await bucket.process(config: serverConfig)
        let bucketServer = BucketServer(bucket: processedBucket)
        let bucketClient = try Client(
            config: serverConfig.clientConfig,
            contexts: processedBucket.clusters[0].database.contexts)
        let bucketQuery = try bucketClient.generateQuery(for: queryVectors, using: secretKey)
        let clusterResponses = try await bucketServer.computeResponses(to: bucketQuery, using: evalKey)

        #expect(clusterResponses.count == 1)
        #expect(clusterResponses[0].clusterId == 0)

        let bucketDistances = try bucketClient.decrypt(
            response: clusterResponses[0].response,
            using: secretKey)

        #expect(directDistances.distances.rowCount == bucketDistances.distances.rowCount)
        #expect(directDistances.distances.columnCount == bucketDistances.distances.columnCount)
        for i in 0..<directDistances.distances.data.count {
            let diff = abs(directDistances.distances.data[i] - bucketDistances.distances.data[i])
            #expect(
                diff < 0.05,
                Comment(
                    rawValue: "Distance mismatch at \(i): " + "direct=\(directDistances.distances.data[i]), "
                        + "bucket=\(bucketDistances.distances.data[i])"))
        }
    }

    @Test
    func multiClusterRoundTrip() async throws {
        let (_, serverConfig) = try Self.makeConfig()

        // 3 clusters, 64 rows total, with some overlap
        let allRows = Self.syntheticRows(count: Self.rowCount)
        let cluster0Rows = Array(allRows[0..<30])
        let cluster1Rows = Array(allRows[20..<50]) // overlaps with cluster 0 at [20..<30]
        let cluster2Rows = Array(allRows[40..<64]) // overlaps with cluster 1 at [40..<50]

        let bucket = Bucket(clusters: [
            ClusterDatabase(clusterId: 0, database: Database(rows: cluster0Rows)),
            ClusterDatabase(clusterId: 1, database: Database(rows: cluster1Rows)),
            ClusterDatabase(clusterId: 2, database: Database(rows: cluster2Rows)),
        ])

        let processedBucket = try await bucket.process(config: serverConfig)
        let bucketServer = BucketServer(bucket: processedBucket)
        let client = try Client(
            config: serverConfig.clientConfig,
            contexts: processedBucket.clusters[0].database.contexts)
        let secretKey = try client.generateSecretKey()
        let evalKey = try client.generateEvaluationKey(using: secretKey)
        let queryVectors = Array2d(data: [allRows[0].vector])
        let query = try client.generateQuery(for: queryVectors, using: secretKey)

        let clusterResponses = try await bucketServer.computeResponses(to: query, using: evalKey)
        #expect(clusterResponses.count == 3)

        // Verify each cluster's distances against plaintext cosine similarity
        let queryNormalized = Self.normalizeRows([allRows[0].vector])[0]
        let clusterRowSets: [[DatabaseRow]] = [cluster0Rows, cluster1Rows, cluster2Rows]

        for clusterResponse in clusterResponses {
            let distances = try client.decrypt(response: clusterResponse.response, using: secretKey)
            let clusterRows = clusterRowSets[Int(clusterResponse.clusterId)]
            let normalizedDB = Self.normalizeRows(clusterRows.map(\.vector))
            let expectedDistances = normalizedDB.map { Self.dot(queryNormalized, $0) }

            #expect(distances.distances.rowCount == clusterRows.count)
            for i in 0..<expectedDistances.count {
                let diff = abs(distances.distances.data[i] - expectedDistances[i])
                #expect(
                    diff < 0.1,
                    Comment(
                        rawValue: "Cluster \(clusterResponse.clusterId) distance mismatch at \(i): "
                            + "got=\(distances.distances.data[i]), expected=\(expectedDistances[i])"))
            }
        }
    }

    @Test
    func duplicateDataProducesIdenticalDistances() async throws {
        let (_, serverConfig) = try Self.makeConfig()
        let rows = Self.syntheticRows(count: Self.rowCount)
        let database = Database(rows: rows)

        let bucket = Bucket(clusters: [
            ClusterDatabase(clusterId: 10, database: database),
            ClusterDatabase(clusterId: 20, database: database),
            ClusterDatabase(clusterId: 30, database: database),
        ])

        let processedBucket = try await bucket.process(config: serverConfig)
        let bucketServer = BucketServer(bucket: processedBucket)
        let client = try Client(
            config: serverConfig.clientConfig,
            contexts: processedBucket.clusters[0].database.contexts)
        let secretKey = try client.generateSecretKey()
        let evalKey = try client.generateEvaluationKey(using: secretKey)
        let queryVectors = Array2d(data: [rows[0].vector])
        let query = try client.generateQuery(for: queryVectors, using: secretKey)

        let clusterResponses = try await bucketServer.computeResponses(to: query, using: evalKey)
        #expect(clusterResponses.count == 3)

        let allDistances = try clusterResponses.map { clusterResponse in
            try client.decrypt(response: clusterResponse.response, using: secretKey)
        }

        // All clusters should produce the same distances
        for i in 1..<allDistances.count {
            #expect(allDistances[i].distances.rowCount == allDistances[0].distances.rowCount)
            for j in 0..<allDistances[0].distances.data.count {
                let diff = abs(allDistances[i].distances.data[j] - allDistances[0].distances.data[j])
                #expect(
                    diff < 0.01,
                    Comment(
                        rawValue: "Cluster \(i) differs from cluster 0 at \(j): "
                            + "\(allDistances[i].distances.data[j]) vs \(allDistances[0].distances.data[j])"))
            }
        }
    }

    @Test
    func bucketWithEmptyCluster() async throws {
        let (_, serverConfig) = try Self.makeConfig()
        let rows = Self.syntheticRows(count: Self.rowCount)

        let bucket = Bucket(clusters: [
            ClusterDatabase(clusterId: 0, database: Database(rows: rows)),
            ClusterDatabase(clusterId: 1, database: Database(rows: [])),
        ])

        let processedBucket = try await bucket.process(config: serverConfig)
        // Empty cluster should be filtered out during processing
        #expect(processedBucket.clusters.count == 1)
        #expect(processedBucket.clusters[0].clusterId == 0)

        let bucketServer = BucketServer(bucket: processedBucket)
        let client = try Client(
            config: serverConfig.clientConfig,
            contexts: processedBucket.clusters[0].database.contexts)
        let secretKey = try client.generateSecretKey()
        let evalKey = try client.generateEvaluationKey(using: secretKey)
        let queryVectors = Array2d(data: [rows[0].vector])
        let query = try client.generateQuery(for: queryVectors, using: secretKey)

        let clusterResponses = try await bucketServer.computeResponses(to: query, using: evalKey)
        #expect(clusterResponses.count == 1)
        #expect(clusterResponses[0].clusterId == 0)

        let distances = try client.decrypt(response: clusterResponses[0].response, using: secretKey)
        #expect(distances.distances.rowCount == Self.rowCount)
    }
}
