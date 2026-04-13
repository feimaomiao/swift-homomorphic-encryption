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

import ApplicationProtobuf
import ArgumentParser
import CTPrivateNearestNeighbourSearch
import Foundation
import HomomorphicEncryption
import HomomorphicEncryptionProtobuf
import Logging
import PrivateNearestNeighborSearch

extension Database {
    init(from path: String) throws {
        let proto = try Apple_SwiftHomomorphicEncryption_Pnns_V1_Database(from: path)
        self = proto.native()
    }
}

/// Validates that `path` has the expected protobuf-serialization extension (`.txtpb` or
/// `.binpb`). Call this on every output path before use so a misconfigured config file cannot
/// redirect writes to arbitrary filesystem locations like `/etc/crontab`.
private func validateProtoFilename(_ path: String, descriptor: String) throws {
    guard path.hasSuffix(".txtpb") || path.hasSuffix(".binpb") else {
        throw ValidationError(
            "'\(descriptor)' must have extension '.txtpb' or '.binpb', found \(path)")
    }
}

/// JSON-encoded arguments for CTPNNSProcessDatabase.
///
/// Output file paths are mandatory so the tool never guesses a destination. A typical config:
/// ```
/// {
///   "inputDatabase": "/in/database.txtpb",
///   "outputEncryptedDatabasePrefix": "/out/encrypted-database",
///   "outputServerConfig": "/out/server-config.txtpb",
///   "outputMetadata": "/out/metadata.binpb",
///   "outputSecretKey": "/out/secret-key.binpb",
///   "outputEvaluationKey": "/out/evaluation-key.binpb",
///   "rlweParameters": "n_4096_logq_27_28_28_logt_16",
///   "distanceMetric": "cosineSimilarity",
///   "batchSize": 1,
///   "scalingFactor": null,
///   "databasePacking": null,
///   "queryPacking": null,
///   "extraPlaintextModuli": null,
///   "trials": 1,
///   "trialDistanceTolerance": 0.05
/// }
/// ```
struct Arguments: Codable, Equatable, Hashable {
    static let defaultArguments = Arguments(
        inputDatabase: "/path/to/input/database.txtpb",
        outputEncryptedDatabasePrefix: "/path/to/output/encrypted-database",
        outputServerConfig: "/path/to/output/server-config.txtpb",
        outputMetadata: "/path/to/output/metadata.binpb",
        outputSecretKey: "/path/to/output/secret-key.binpb",
        outputEvaluationKey: "/path/to/output/evaluation-key.binpb",
        rlweParameters: .n_8192_logq_3x55_logt_30,
        extraPlaintextModuli: nil,
        distanceMetric: .cosineSimilarity,
        batchSize: 1,
        scalingFactor: nil,
        databasePacking: nil,
        queryPacking: nil,
        trials: 1,
        trialDistanceTolerance: 0.01)

    let inputDatabase: String
    /// File name prefix for emitted `SerializedCiphertextMatrix` files.
    ///
    /// The tool emits `<prefix>-<i>.binpb` for each plaintext modulus index `i`. With a single
    /// plaintext modulus (the common case) there's exactly one ciphertext file: `<prefix>-0.binpb`.
    let outputEncryptedDatabasePrefix: String
    let outputServerConfig: String
    /// Empty-plaintextMatrices `SerializedProcessedDatabase` carrying `entryIds`,
    /// `entryMetadatas`, and `serverConfig` — the side-channel metadata the server needs to
    /// return identifiers alongside scores.
    let outputMetadata: String
    /// Sensitive. Kept by the client only; never ship this to the server.
    let outputSecretKey: String
    /// Contains the BSGS Galois elements and a relinearization key. Ship this to the server.
    let outputEvaluationKey: String

    let rlweParameters: PredefinedRlweParameters
    let extraPlaintextModuli: [UInt64]?
    let distanceMetric: DistanceMetric?
    let batchSize: Int?
    let scalingFactor: Int?
    let databasePacking: MatrixPacking?
    let queryPacking: MatrixPacking?
    let trials: Int?
    let trialDistanceTolerance: Float?

    static func defaultJsonString(vectorDimension: Int) -> String {
        // Called from `--help` text and is strictly informational — never crash the CLI if the
        // default arguments happen to not resolve for an unusual RLWE preset.
        guard let resolved = try? defaultArguments.resolve(
            for: vectorDimension, scheme: Bfv<UInt64>.self)
        else {
            return "<default configuration unavailable for vectorDimension=\(vectorDimension)>"
        }
        let defaults = Arguments(
            inputDatabase: resolved.inputDatabase,
            outputEncryptedDatabasePrefix: resolved.outputEncryptedDatabasePrefix,
            outputServerConfig: resolved.outputServerConfig,
            outputMetadata: resolved.outputMetadata,
            outputSecretKey: resolved.outputSecretKey,
            outputEvaluationKey: resolved.outputEvaluationKey,
            rlweParameters: resolved.rlweParameters,
            extraPlaintextModuli: resolved.extraPlaintextModuli,
            distanceMetric: resolved.distanceMetric,
            batchSize: resolved.batchSize,
            scalingFactor: resolved.scalingFactor,
            databasePacking: resolved.databasePacking,
            queryPacking: resolved.queryPacking,
            trials: resolved.trials,
            trialDistanceTolerance: resolved.trialDistanceTolerance)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        guard let data = try? encoder.encode(defaults) else {
            return "<default configuration failed to encode>"
        }
        // swiftlint:disable:next optional_data_string_conversion
        return String(decoding: data, as: UTF8.self)
    }

    func resolve<Scheme: HeScheme>(for vectorDimension: Int, scheme _: Scheme.Type) throws
        -> ResolvedArguments
    {
        // Reject paths that don't look like protobuf artifacts before the tool writes
        // anything to disk. `outputEncryptedDatabasePrefix` is a bare prefix (no extension)
        // because the tool appends "-<i>.binpb" per plaintext modulus, so it's not validated
        // here — the final path includes the suffix and will satisfy readers downstream.
        try validateProtoFilename(inputDatabase, descriptor: "inputDatabase")
        try validateProtoFilename(outputServerConfig, descriptor: "outputServerConfig")
        try validateProtoFilename(outputMetadata, descriptor: "outputMetadata")
        try validateProtoFilename(outputSecretKey, descriptor: "outputSecretKey")
        try validateProtoFilename(outputEvaluationKey, descriptor: "outputEvaluationKey")

        let distanceMetric = distanceMetric ?? .cosineSimilarity
        let databasePacking = databasePacking ?? .diagonal(
            babyStepGiantStep: BabyStepGiantStep(vectorDimension: vectorDimension))
        let queryPacking = queryPacking ?? .denseRow
        let plaintextModuli = [rlweParameters.plaintextModulus] + (extraPlaintextModuli ?? [])
        let scalingFactor = scalingFactor ?? ClientConfig<Scheme>.maxScalingFactor(
            distanceMetric: distanceMetric,
            vectorDimension: vectorDimension,
            plaintextModuli: plaintextModuli.map { Scheme.Scalar($0) })
        return ResolvedArguments(
            inputDatabase: inputDatabase,
            outputEncryptedDatabasePrefix: outputEncryptedDatabasePrefix,
            outputServerConfig: outputServerConfig,
            outputMetadata: outputMetadata,
            outputSecretKey: outputSecretKey,
            outputEvaluationKey: outputEvaluationKey,
            rlweParameters: rlweParameters,
            extraPlaintextModuli: extraPlaintextModuli ?? [],
            distanceMetric: distanceMetric,
            batchSize: batchSize ?? 1,
            scalingFactor: scalingFactor,
            databasePacking: databasePacking,
            queryPacking: queryPacking,
            trials: trials ?? 1,
            trialDistanceTolerance: trialDistanceTolerance ?? 0.01)
    }
}

struct ResolvedArguments: CustomStringConvertible, Encodable {
    let inputDatabase: String
    let outputEncryptedDatabasePrefix: String
    let outputServerConfig: String
    let outputMetadata: String
    let outputSecretKey: String
    let outputEvaluationKey: String
    let rlweParameters: PredefinedRlweParameters
    let extraPlaintextModuli: [UInt64]
    let distanceMetric: DistanceMetric
    let batchSize: Int
    let scalingFactor: Int
    let databasePacking: MatrixPacking
    let queryPacking: MatrixPacking
    let trials: Int
    let trialDistanceTolerance: Float

    var description: String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        // `description` is called inside logger interpolation on every run; a JSON encoding
        // failure should never crash the CLI — fall back to a minimal summary instead.
        guard let data = try? encoder.encode(self) else {
            return "ResolvedArguments(input=\(inputDatabase), rlwe=\(rlweParameters), metric=\(distanceMetric))"
        }
        // swiftlint:disable:next optional_data_string_conversion
        return String(decoding: data, as: UTF8.self)
    }
}

@main
struct ProcessDatabase: AsyncParsableCommand {
    private struct TrialInputs<Scheme: HeScheme> {
        let database: Database
        let encryptedMatrices: [CiphertextMatrix<Scheme, Scheme.CanonicalCiphertextFormat>]
        let contexts: [Scheme.Context]
        let clientConfig: ClientConfig<Scheme>
        let secretKey: SecretKey<Scheme>
        let evaluationKey: EvaluationKey<Scheme>
        let resolved: ResolvedArguments
    }

    private struct WriteOutputsInputs<Scheme: HeScheme> {
        let resolved: ResolvedArguments
        let database: Database
        let serverConfig: ServerConfig<Scheme>
        let encryptedMatrices: [CiphertextMatrix<Scheme, Scheme.CanonicalCiphertextFormat>]
        let secretKey: SecretKey<Scheme>
        let evaluationKey: EvaluationKey<Scheme>
    }

    static let configuration = CommandConfiguration(
        commandName: "CTPNNSProcessDatabase",
        abstract: """
            Ciphertext × ciphertext PNNS database processor — turns a .txtpb database into an \
            encrypted .binpb bundle that an offshore server can evaluate queries against without \
            holding the secret key.
            """)

    static let logger = Logger(label: "CTPNNSProcessDatabase")

    @Argument(
        help: """
            Path to a JSON configuration file.

            A default JSON config (vectorDimension=128) is:
            \(Arguments.defaultJsonString(vectorDimension: 128))
            """)
    var configFile: String

    mutating func run() async throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: configFile))
        let args = try JSONDecoder().decode(Arguments.self, from: data)
        if args.rlweParameters.supportsScalar(UInt32.self) {
            try await process(args: args, scheme: Bfv<UInt32>.self)
        } else {
            try await process(args: args, scheme: Bfv<UInt64>.self)
        }
    }

    mutating func process<Scheme: HeScheme>(args: Arguments, scheme _: Scheme.Type) async throws {
        // --- Read + quantize ---
        let database = try Database(from: args.inputDatabase)
        guard let vectorDimension = database.rows.first?.vector.count else {
            throw PnnsError.emptyDatabase
        }
        let plaintextMatrixDimensions = try MatrixDimensions(
            rowCount: database.rows.count, columnCount: vectorDimension)

        let resolved = try args.resolve(for: vectorDimension, scheme: Scheme.self)
        ProcessDatabase.logger.info("Processing database with configuration: \(resolved)")

        let encryptionParameters = try EncryptionParameters<Scheme.Scalar>(from: resolved.rlweParameters)
        let evaluationKeyConfig = try CTMatrixMultiplication.evaluationKeyConfig(
            plaintextMatrixDimensions: plaintextMatrixDimensions,
            maxQueryCount: resolved.batchSize,
            encryptionParameters: encryptionParameters,
            scheme: Scheme.self)
        let clientConfig = try ClientConfig<Scheme>(
            encryptionParameters: encryptionParameters,
            scalingFactor: resolved.scalingFactor,
            queryPacking: resolved.queryPacking,
            vectorDimension: vectorDimension,
            evaluationKeyConfig: evaluationKeyConfig,
            distanceMetric: resolved.distanceMetric,
            extraPlaintextModuli: resolved.extraPlaintextModuli.map { Scheme.Scalar($0) })
        let serverConfig = ServerConfig<Scheme>(
            clientConfig: clientConfig,
            databasePacking: resolved.databasePacking)

        // One context per plaintext modulus.
        let contexts = try clientConfig.encryptionParameters.map { params in
            try Scheme.Context(encryptionParameters: params)
        }
        let secretKey = try contexts[0].generateSecretKey()
        let evaluationKey = try contexts[0].generateEvaluationKey(
            config: evaluationKeyConfig, using: secretKey)

        // --- Normalize + scale + round (identical to Database.process) ---
        let vectors = Array2d(data: database.rows.map(\.vector))
        let roundedVectors: Array2d<Scheme.SignedScalar> = switch resolved.distanceMetric {
        case .cosineSimilarity:
            vectors.normalizedScaledAndRounded(scalingFactor: Float(resolved.scalingFactor))
        case .dotProduct:
            vectors.scaled(by: Float(resolved.scalingFactor)).rounded()
        }

        // --- Pack (Coeff) + encrypt, one matrix per plaintext modulus ---
        let dims = try MatrixDimensions(roundedVectors.shape)
        let shouldReduce = contexts.count > 1
        let encryptedMatrices = try contexts.map { context in
            try PlaintextMatrix<Scheme, Coeff>(
                context: context,
                dimensions: dims,
                packing: resolved.databasePacking,
                signedValues: roundedVectors.data,
                reduce: shouldReduce).encrypt(using: secretKey)
        }
        ProcessDatabase.logger.info(
            "Encrypted \(encryptedMatrices.count) plaintext-modulus matrix(es) under a fresh secret key")

        // --- Optional trial: self-similarity of the first `batchSize` rows should ≈ 1 ---
        if resolved.trials > 0 {
            try await runTrial(inputs: TrialInputs(
                database: database,
                encryptedMatrices: encryptedMatrices,
                contexts: contexts,
                clientConfig: clientConfig,
                secretKey: secretKey,
                evaluationKey: evaluationKey,
                resolved: resolved))
        }

        // --- Write outputs ---
        try writeOutputs(inputs: WriteOutputsInputs(
            resolved: resolved,
            database: database,
            serverConfig: serverConfig,
            encryptedMatrices: encryptedMatrices,
            secretKey: secretKey,
            evaluationKey: evaluationKey))
    }

    private func runTrial<Scheme: HeScheme>(inputs: TrialInputs<Scheme>) async throws {
        let database = inputs.database
        let encryptedMatrices = inputs.encryptedMatrices
        let contexts = inputs.contexts
        let clientConfig = inputs.clientConfig
        let secretKey = inputs.secretKey
        let evaluationKey = inputs.evaluationKey
        let resolved = inputs.resolved
        ProcessDatabase.logger.info("Running trial: self-similarity of the first \(resolved.batchSize) row(s)")
        // Use the first batchSize database rows as queries (self-similarity → should be ≈ 1).
        let queryVector = database.rows[0].vector
        let queryFloat = Array2d(data: queryVector, rowCount: 1, columnCount: queryVector.count)
        let queryScaled: Array2d<Scheme.SignedScalar> = switch resolved.distanceMetric {
        case .cosineSimilarity:
            queryFloat.normalizedScaledAndRounded(scalingFactor: Float(resolved.scalingFactor))
        case .dotProduct:
            queryFloat.scaled(by: Float(resolved.scalingFactor)).rounded()
        }
        let queryDims = try MatrixDimensions(rowCount: 1, columnCount: queryVector.count)
        let queryContext = contexts[0]
        let queryCt = try PlaintextMatrix<Scheme, Coeff>(
            context: queryContext,
            dimensions: queryDims,
            packing: clientConfig.queryPacking,
            signedValues: queryScaled.data,
            reduce: contexts.count > 1).encrypt(using: secretKey)

        // Run the CT kernel against the first plaintext modulus.
        let responseCts = try await encryptedMatrices[0].mulTranspose(
            vector: queryCt, using: evaluationKey)

        let noiseBudget = try responseCts.map { ciphertext in
            try ciphertext.noiseBudget(using: secretKey, variableTime: true)
        }.min() ?? -.infinity
        ProcessDatabase.logger.info("Trial noise budget = \(noiseBudget) bits")

        // Decrypt, de-scale, and check the first row's self-similarity.
        var decoded: [Scheme.Scalar] = []
        for ct in responseCts {
            decoded += try ct.decrypt(using: secretKey).decode(format: .simd) as [Scheme.Scalar]
        }
        guard let firstSlot = decoded.first else {
            throw ValidationError(
                "Trial produced no decoded slots — response had \(responseCts.count) ciphertext(s)")
        }
        let p = queryContext.plaintextModulus
        let s2 = Float(resolved.scalingFactor) * Float(resolved.scalingFactor)
        let selfSimilarity = Float(firstSlot.remainderToCentered(modulus: p)) / s2

        let error = abs(selfSimilarity - 1.0)
        ProcessDatabase.logger.info(
            """
            Trial row 0 self-similarity = \(selfSimilarity), error = \(error) \
            (tolerance \(resolved.trialDistanceTolerance))
            """)
        guard error <= resolved.trialDistanceTolerance else {
            throw ValidationError(
                "Trial error \(error) exceeds tolerance \(resolved.trialDistanceTolerance)")
        }
    }

    private func writeOutputs<Scheme: HeScheme>(inputs: WriteOutputsInputs<Scheme>) throws {
        let resolved = inputs.resolved
        let database = inputs.database
        let serverConfig = inputs.serverConfig
        let encryptedMatrices = inputs.encryptedMatrices
        let secretKey = inputs.secretKey
        let evaluationKey = inputs.evaluationKey
        // 1. Server config.
        try serverConfig.proto().save(to: resolved.outputServerConfig)
        ProcessDatabase.logger.info("Saved server config to \(resolved.outputServerConfig)")

        // 2. Metadata bundle: entryIds, entryMetadatas, serverConfig
        //    (reuse SerializedProcessedDatabase with plaintextMatrices left empty).
        let entryIds = database.rows.map(\.entryId)
        let hasMetadata = database.rows.contains { !$0.entryMetadata.isEmpty }
        let entryMetadatas = hasMetadata ? database.rows.map(\.entryMetadata) : []
        var metaProto = Apple_SwiftHomomorphicEncryption_Pnns_V1_SerializedProcessedDatabase()
        metaProto.plaintextMatrices = []
        metaProto.entryIds = entryIds
        metaProto.entryMetadatas = entryMetadatas.map { Data($0) }
        metaProto.serverConfig = try serverConfig.proto()
        try metaProto.save(to: resolved.outputMetadata)
        ProcessDatabase.logger.info("Saved metadata bundle to \(resolved.outputMetadata)")

        // 3. Ciphertext matrices, one file per plaintext modulus.
        for (index, matrix) in encryptedMatrices.enumerated() {
            let serialized: SerializedCiphertextMatrix<Scheme.Scalar> = try matrix.serialize()
            let proto = try serialized.proto()
            let path = "\(resolved.outputEncryptedDatabasePrefix)-\(index).binpb"
            try proto.save(to: path)
            let size = (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int) ?? -1
            ProcessDatabase.logger.info(
                "Saved encrypted matrix \(index) to \(path) (\(size) bytes)")
        }

        // 4. Secret key (SENSITIVE — client only).
        let secretKeyProto = secretKey.serialize().proto()
        try secretKeyProto.save(to: resolved.outputSecretKey)
        ProcessDatabase.logger.warning(
            "Saved secret key to \(resolved.outputSecretKey) — this file is sensitive; do NOT ship to the server.")

        // 5. Evaluation key (ship to server).
        let evalKeyProto = evaluationKey.serialize().proto()
        try evalKeyProto.save(to: resolved.outputEvaluationKey)
        ProcessDatabase.logger.info(
            "Saved evaluation key to \(resolved.outputEvaluationKey) — ship this with the encrypted database.")
    }
}
