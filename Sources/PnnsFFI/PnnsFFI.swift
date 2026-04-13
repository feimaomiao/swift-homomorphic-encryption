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

// PnnsFFI — C-callable wrapper around Apple's swift-homomorphic-encryption PNNS.
//
// Exposes keygen, encrypt, create_server, compute, decrypt for Bfv<UInt32> and Bfv<UInt64>.
// HE objects (Query, Response, EvalKey) stay as opaque handles — Rust holds pointers
// but never inspects the ciphertext internals.

// swiftlint:disable function_parameter_count missing_docs

import ApplicationProtobuf
import CTPrivateNearestNeighbourSearch
import Foundation
import HomomorphicEncryption
import HomomorphicEncryptionProtobuf
import PrivateNearestNeighborSearch
import SwiftProtobuf

// Disambiguate the two `Server` types in this file.
private typealias PTServer = PrivateNearestNeighborSearch.Server
private typealias CTServer = CTPrivateNearestNeighbourSearch.Server

// MARK: - Byte buffer for returning variable-length data to Rust

public struct ByteBuf {
    public var ptr: UnsafeMutablePointer<UInt8>?
    public var len: Int
}

// MARK: - Opaque handle helpers

private final class AnyBox {
    let value: Any
    init(_ value: Any) {
        self.value = value
    }
}

private func box(_ value: Any) -> UnsafeMutableRawPointer {
    Unmanaged.passRetained(AnyBox(value)).toOpaque()
}

private func unbox<T>(_ ptr: UnsafeRawPointer, as _: T.Type) -> T {
    guard let value = Unmanaged<AnyBox>.fromOpaque(ptr).takeUnretainedValue().value as? T else {
        preconditionFailure("PnnsFFI: handle type mismatch — expected \(T.self)")
    }
    return value
}

private func release(_ ptr: UnsafeMutableRawPointer) {
    Unmanaged<AnyBox>.fromOpaque(ptr).release()
}

// MARK: - Internal context holding concrete scheme types

private struct PnnsCtx<Scheme: HeScheme>: @unchecked Sendable {
    let params: EncryptionParameters<Scheme.Scalar>
    let context: Scheme.Context
    let clientConfig: ClientConfig<Scheme>
    let serverConfig: ServerConfig<Scheme>
    let client: Client<Scheme>
}

/// Run an async closure synchronously (blocking the current thread).
/// Used at the FFI boundary where we must return a value to C.
/// Safety: the semaphore guarantees the Task completes before we read `result`.
private func runBlocking<T>(_ body: @escaping () async throws -> T) -> T? {
    let holder = UnsafeMutablePointer<T?>.allocate(capacity: 1)
    holder.initialize(to: nil)
    let errHolder = UnsafeMutablePointer<(any Error)?>.allocate(capacity: 1)
    errHolder.initialize(to: nil)
    let semaphore = DispatchSemaphore(value: 0)

    let sendableHolder = SendablePointer(holder)
    let sendableErr = SendablePointer(errHolder)
    let sendableSem = SendableValue(semaphore)
    let sendableBody = SendableValue(body)

    Task { @Sendable in
        do {
            sendableHolder.ptr.pointee = try await sendableBody.value()
        } catch {
            sendableErr.ptr.pointee = error
        }
        sendableSem.value.signal()
    }
    semaphore.wait()

    let result = holder.pointee
    let error = errHolder.pointee
    holder.deinitialize(count: 1); holder.deallocate()
    errHolder.deinitialize(count: 1); errHolder.deallocate()

    if let error { print("[pnns] runBlocking error: \(error)") }
    return result
}

/// Wrapper to make a raw pointer Sendable for cross-Task use.
private struct SendablePointer<T>: @unchecked Sendable {
    let ptr: UnsafeMutablePointer<T>
    init(_ ptr: UnsafeMutablePointer<T>) {
        self.ptr = ptr
    }
}

private struct SendableValue<T>: @unchecked Sendable {
    let value: T
    init(_ value: T) {
        self.value = value
    }
}

private func makeCtx<Scheme: HeScheme>(
    _: Scheme.Type,
    rlweParams: PredefinedRlweParameters,
    vectorDimension: Int,
    databaseRowCount: Int) throws -> PnnsCtx<Scheme>
{
    let params = try EncryptionParameters<Scheme.Scalar>(from: rlweParams)
    let context = try Scheme.Context(encryptionParameters: params)

    let plaintextModuli = [params.plaintextModulus]
    let scalingFactor = ClientConfig<Scheme>.maxScalingFactor(
        distanceMetric: .cosineSimilarity,
        vectorDimension: vectorDimension,
        plaintextModuli: plaintextModuli)

    let dbDimensions = try MatrixDimensions(
        rowCount: databaseRowCount,
        columnCount: vectorDimension)
    let evalKeyConfig = try CosineSimilarity.evaluationKeyConfig(
        plaintextMatrixDimensions: dbDimensions,
        maxQueryCount: 1,
        encryptionParameters: params,
        scheme: Scheme.self)

    let clientConfig = try ClientConfig<Scheme>(
        encryptionParameters: params,
        scalingFactor: scalingFactor,
        queryPacking: .denseRow,
        vectorDimension: vectorDimension,
        evaluationKeyConfig: evalKeyConfig,
        distanceMetric: .cosineSimilarity)

    let serverConfig = ServerConfig<Scheme>(
        clientConfig: clientConfig,
        databasePacking: .diagonal(
            babyStepGiantStep: BabyStepGiantStep(vectorDimension: vectorDimension)))

    let client = try Client<Scheme>(config: clientConfig, contexts: [context])
    return PnnsCtx(
        params: params,
        context: context,
        clientConfig: clientConfig,
        serverConfig: serverConfig,
        client: client)
}

// MARK: - Scalar detection

/// Returns 32 if the default RLWE parameters support UInt32 (faster),
/// or 64 if only UInt64 is supported.
/// The RLWE parameters are currently hardcoded to n_4096_logq_27_28_28_logt_17.
@_cdecl("pnns_detect_scalar")
public func pnnsDetectScalar() -> Int32 {
    let params = PredefinedRlweParameters.n_4096_logq_27_28_28_logt_17
    return params.supportsScalar(UInt32.self) ? 32 : 64
}

// MARK: - Bfv<UInt64> FFI

/// Create context. `vector_dim` is the embedding dimension, `db_rows` is number of database vectors.
/// Returns opaque handle or NULL.
@_cdecl("pnns64_create_context")
public func pnns64CreateContext(vectorDim: Int32, dbRows: Int32) -> UnsafeMutableRawPointer? {
    do {
        let ctx = try makeCtx(
            Bfv<UInt64>.self,
            rlweParams: .n_4096_logq_27_28_28_logt_17,
            vectorDimension: Int(vectorDim),
            databaseRowCount: Int(dbRows))
        return box(ctx)
    } catch {
        print("[pnns64] create_context error: \(error)")
        return nil
    }
}

/// Generate a secret key. Returns opaque handle.
@_cdecl("pnns64_keygen")
public func pnns64Keygen(ctxPtr: UnsafeRawPointer) -> UnsafeMutableRawPointer? {
    let ctx = unbox(ctxPtr, as: PnnsCtx<Bfv<UInt64>>.self)
    do {
        let sk = try ctx.client.generateSecretKey()
        return box(sk)
    } catch {
        print("[pnns64] keygen error: \(error)")
        return nil
    }
}

/// Generate an evaluation key. Returns opaque handle.
@_cdecl("pnns64_gen_eval_key")
public func pnns64GenEvalKey(
    ctxPtr: UnsafeRawPointer,
    skPtr: UnsafeRawPointer) -> UnsafeMutableRawPointer?
{
    let ctx = unbox(ctxPtr, as: PnnsCtx<Bfv<UInt64>>.self)
    let sk = unbox(skPtr, as: SecretKey<Bfv<UInt64>>.self)
    do {
        let evalKey = try ctx.client.generateEvaluationKey(using: sk)
        return box(evalKey)
    } catch {
        print("[pnns64] gen_eval_key error: \(error)")
        return nil
    }
}

/// Encrypt a query vector. `vector` points to `dim` floats. Returns opaque Query handle.
@_cdecl("pnns64_encrypt")
public func pnns64Encrypt(
    ctxPtr: UnsafeRawPointer,
    skPtr: UnsafeRawPointer,
    vector: UnsafePointer<Float>,
    dim: Int32) -> UnsafeMutableRawPointer?
{
    let ctx = unbox(ctxPtr, as: PnnsCtx<Bfv<UInt64>>.self)
    let sk = unbox(skPtr, as: SecretKey<Bfv<UInt64>>.self)
    do {
        let floats = Array(UnsafeBufferPointer(start: vector, count: Int(dim)))
        let vectors = Array2d(data: floats, rowCount: 1, columnCount: Int(dim))
        let query = try ctx.client.generateQuery(for: vectors, using: sk)
        return box(query)
    } catch {
        print("[pnns64] encrypt error: \(error)")
        return nil
    }
}

/// Create a server from a database of float vectors.
/// `db_vectors` is row-major: db_rows × vector_dim floats.
/// Returns opaque Server handle. This processes the database (may take a moment).
@_cdecl("pnns64_create_server")
public func pnns64CreateServer(
    ctxPtr: UnsafeRawPointer,
    dbVectors: UnsafePointer<Float>,
    dbRows: Int32,
    vectorDim: Int32) -> UnsafeMutableRawPointer?
{
    let ctx = unbox(ctxPtr, as: PnnsCtx<Bfv<UInt64>>.self)
    let count = Int(dbRows) * Int(vectorDim)
    let floats = Array(UnsafeBufferPointer(start: dbVectors, count: count))

    let rows = (0..<Int(dbRows)).map { i in
        let start = i * Int(vectorDim)
        let end = start + Int(vectorDim)
        return DatabaseRow(
            entryId: UInt64(i),
            entryMetadata: [],
            vector: Array(floats[start..<end]))
    }
    let database = Database(rows: rows)
    let config = ctx.serverConfig

    guard let server: PTServer<Bfv<UInt64>> = runBlocking({
        let processed = try await database.process(config: config)
        return try PTServer(database: processed)
    }) else { return nil }
    return box(server)
}

/// Compute the server's response to a query. Returns opaque Response handle.
/// `modSwitchDown` (0 = false, nonzero = true) controls whether the server
/// shrinks the response ciphertext modulus. Pass 0 when the caller intends
/// to homomorphically aggregate responses (e.g. LB-side) before the final
/// mod-switch.
@_cdecl("pnns64_compute")
public func pnns64Compute(
    serverPtr: UnsafeRawPointer,
    queryPtr: UnsafeRawPointer,
    evalKeyPtr: UnsafeRawPointer,
    modSwitchDown: Int32) -> UnsafeMutableRawPointer?
{
    let server = unbox(serverPtr, as: PTServer<Bfv<UInt64>>.self)
    let query = unbox(queryPtr, as: Query<Bfv<UInt64>>.self)
    let evalKey = unbox(evalKeyPtr, as: EvaluationKey<Bfv<UInt64>>.self)
    let mod = modSwitchDown != 0

    guard let response: Response<Bfv<UInt64>> = runBlocking({
        try await server.computeResponse(to: query, using: evalKey, modSwitchDown: mod)
    }) else { return nil }
    return box(response)
}

/// Decrypt a response. Writes distances into caller-provided buffer.
/// `out_distances` must have space for db_rows floats.
/// `out_entry_ids` must have space for db_rows UInt64s.
/// Returns the number of database rows, or -1 on error.
@_cdecl("pnns64_decrypt")
public func pnns64Decrypt(
    ctxPtr: UnsafeRawPointer,
    skPtr: UnsafeRawPointer,
    responsePtr: UnsafeRawPointer,
    outDistances: UnsafeMutablePointer<Float>,
    outEntryIds: UnsafeMutablePointer<UInt64>,
    outLen: Int32) -> Int32
{
    let ctx = unbox(ctxPtr, as: PnnsCtx<Bfv<UInt64>>.self)
    let sk = unbox(skPtr, as: SecretKey<Bfv<UInt64>>.self)
    let response = unbox(responsePtr, as: Response<Bfv<UInt64>>.self)
    do {
        let result = try ctx.client.decrypt(response: response, using: sk)
        let shape = result.distances.shape
        let totalEntries = shape.rowCount
        guard totalEntries <= Int(outLen) else {
            print("[pnns64] decrypt: buffer too small (\(outLen) < \(totalEntries))")
            return -1
        }
        // Copy distances row by row via public API
        for r in 0..<shape.rowCount {
            // For single-query, columnCount == 1; distance is at [r, 0]
            for c in 0..<shape.columnCount {
                outDistances[r * shape.columnCount + c] = result.distances[r, c]
            }
        }
        for i in 0..<min(result.entryIds.count, Int(outLen)) {
            outEntryIds[i] = result.entryIds[i]
        }
        return Int32(totalEntries)
    } catch {
        print("[pnns64] decrypt error: \(error)")
        return -1
    }
}

/// Save a processed database to a protobuf file.
/// `server_ptr` is the Server handle. `path` is a null-terminated C string.
/// Returns 0 on success, -1 on error.
@_cdecl("pnns64_save_db")
public func pnns64SaveDb(
    serverPtr: UnsafeRawPointer,
    path: UnsafePointer<CChar>) -> Int32
{
    let server = unbox(serverPtr, as: PTServer<Bfv<UInt64>>.self)
    let filePath = String(cString: path)
    do {
        let serialized = try server.database.serialize()
        let proto = try serialized.proto()
        try proto.save(to: filePath)
        return 0
    } catch {
        print("[pnns64] save_db error: \(error)")
        return -1
    }
}

/// Load a processed database from a protobuf file and create a Server.
/// `path` is a null-terminated C string to a .binpb file.
/// Returns opaque Server handle, or NULL on error.
@_cdecl("pnns64_load_server")
public func pnns64LoadServer(
    path: UnsafePointer<CChar>) -> UnsafeMutableRawPointer?
{
    let filePath = String(cString: path)
    do {
        let proto = try Apple_SwiftHomomorphicEncryption_Pnns_V1_SerializedProcessedDatabase(from: filePath)
        let serialized: SerializedProcessedDatabase<Bfv<UInt64>> = try proto.native()
        let processed = try ProcessedDatabase<Bfv<UInt64>>(from: serialized)
        let server = try PTServer(database: processed)
        return box(server)
    } catch {
        print("[pnns64] load_server error: \(error)")
        return nil
    }
}

/// Load a server and also return the client config needed for context creation.
/// Writes the vector_dim and db_rows to the output pointers so Rust can create a matching context.
/// Returns opaque Server handle, or NULL on error.
@_cdecl("pnns64_load_server_with_config")
public func pnns64LoadServerWithConfig(
    path: UnsafePointer<CChar>,
    outVectorDim: UnsafeMutablePointer<Int32>,
    outDbRows: UnsafeMutablePointer<Int32>) -> UnsafeMutableRawPointer?
{
    let filePath = String(cString: path)
    do {
        let proto = try Apple_SwiftHomomorphicEncryption_Pnns_V1_SerializedProcessedDatabase(from: filePath)
        let serialized: SerializedProcessedDatabase<Bfv<UInt64>> = try proto.native()
        let processed = try ProcessedDatabase<Bfv<UInt64>>(from: serialized)
        let server = try PTServer(database: processed)
        outVectorDim.pointee = Int32(server.config.vectorDimension)
        outDbRows.pointee = Int32(server.database.entryIds.count)
        return box(server)
    } catch {
        print("[pnns64] load_server_with_config error: \(error)")
        return nil
    }
}

// MARK: - Bfv<UInt64> Fast Encoder

/// Create a FastPlaintextEncoder for rapid database re-encoding.
/// The encoder caches the slot mapping for a fixed (dbRows, vectorDim) shape.
/// Returns opaque encoder handle, or NULL on error.
@_cdecl("pnns64_create_fast_encoder")
public func pnns64CreateFastEncoder(
    ctxPtr: UnsafeRawPointer,
    dbRows: Int32) -> UnsafeMutableRawPointer?
{
    let ctx = unbox(ctxPtr, as: PnnsCtx<Bfv<UInt64>>.self)
    do {
        let encoder = try FastPlaintextEncoder<Bfv<UInt64>>(
            config: ctx.serverConfig,
            context: ctx.context,
            rowCount: Int(dbRows))
        return box(encoder)
    } catch {
        print("[pnns64] create_fast_encoder error: \(error)")
        return nil
    }
}

/// Quantize float vectors into a compact integer representation.
/// `db_vectors` is row-major: dbRows × vectorDim floats.
/// Writes quantized bytes to `out_bytes` (caller must allocate dbRows * vectorDim * 8 bytes for UInt64).
/// Returns the number of bytes written, or -1 on error.
@_cdecl("pnns64_quantize_db")
public func pnns64QuantizeDb(
    ctxPtr: UnsafeRawPointer,
    dbVectors: UnsafePointer<Float>,
    dbRows: Int32,
    vectorDim: Int32,
    outBytes: UnsafeMutablePointer<UInt8>,
    outBytesCapacity: Int,
    outEntryIds: UnsafeMutablePointer<UInt64>) -> Int
{
    let ctx = unbox(ctxPtr, as: PnnsCtx<Bfv<UInt64>>.self)
    let count = Int(dbRows) * Int(vectorDim)
    let floats = Array(UnsafeBufferPointer(start: dbVectors, count: count))

    let rows = (0..<Int(dbRows)).map { i in
        let start = i * Int(vectorDim)
        let end = start + Int(vectorDim)
        return DatabaseRow(
            entryId: UInt64(i),
            entryMetadata: [],
            vector: Array(floats[start..<end]))
    }
    let database = Database(rows: rows)
    let quantized = QuantizedDatabase<Bfv<UInt64>>(database: database, config: ctx.serverConfig)
    let bytes = quantized.serializeVectors()

    guard bytes.count <= outBytesCapacity else {
        print("[pnns64] quantize_db: buffer too small (\(outBytesCapacity) < \(bytes.count))")
        return -1
    }
    bytes.withUnsafeBufferPointer { buf in
        if let base = buf.baseAddress {
            outBytes.update(from: base, count: buf.count)
        }
    }
    for i in 0..<Int(dbRows) {
        outEntryIds[i] = quantized.entryIds[i]
    }
    return bytes.count
}

/// Create a Server from a quantized database using the fast encoder.
/// `quantized_bytes`/`quantized_len` are from `pnns64_quantize_db`.
/// `entry_ids` are the UInt64 entry IDs (dbRows count).
/// Returns opaque Server handle, or NULL on error.
@_cdecl("pnns64_fast_encode_server")
public func pnns64FastEncodeServer(
    ctxPtr: UnsafeRawPointer,
    encoderPtr: UnsafeRawPointer,
    quantizedBytes: UnsafePointer<UInt8>,
    quantizedLen: Int,
    entryIds: UnsafePointer<UInt64>,
    dbRows: Int32,
    vectorDim: Int32) -> UnsafeMutableRawPointer?
{
    let ctx = unbox(ctxPtr, as: PnnsCtx<Bfv<UInt64>>.self)
    let encoder = unbox(encoderPtr, as: FastPlaintextEncoder<Bfv<UInt64>>.self)
    do {
        let byteArray = Array(UnsafeBufferPointer(start: quantizedBytes, count: quantizedLen))
        let ids = Array(UnsafeBufferPointer(start: entryIds, count: Int(dbRows)))
        let quantizedDB = QuantizedDatabase<Bfv<UInt64>>.deserializeVectors(
            from: byteArray,
            rowCount: Int(dbRows),
            vectorDimension: Int(vectorDim),
            entryIds: ids)
        let processed = try encoder.encodeDatabase(quantizedDB, context: ctx.context)
        let server = try PTServer(database: processed)
        return box(server)
    } catch {
        print("[pnns64] fast_encode_server error: \(error)")
        return nil
    }
}

// MARK: - Bfv<UInt32> FFI

@_cdecl("pnns32_create_context")
public func pnns32CreateContext(vectorDim: Int32, dbRows: Int32) -> UnsafeMutableRawPointer? {
    do {
        let ctx = try makeCtx(
            Bfv<UInt32>.self,
            rlweParams: .n_4096_logq_27_28_28_logt_17,
            vectorDimension: Int(vectorDim),
            databaseRowCount: Int(dbRows))
        return box(ctx)
    } catch {
        print("[pnns32] create_context error: \(error)")
        return nil
    }
}

@_cdecl("pnns32_keygen")
public func pnns32Keygen(ctxPtr: UnsafeRawPointer) -> UnsafeMutableRawPointer? {
    let ctx = unbox(ctxPtr, as: PnnsCtx<Bfv<UInt32>>.self)
    do {
        let sk = try ctx.client.generateSecretKey()
        return box(sk)
    } catch {
        print("[pnns32] keygen error: \(error)")
        return nil
    }
}

@_cdecl("pnns32_gen_eval_key")
public func pnns32GenEvalKey(
    ctxPtr: UnsafeRawPointer,
    skPtr: UnsafeRawPointer) -> UnsafeMutableRawPointer?
{
    let ctx = unbox(ctxPtr, as: PnnsCtx<Bfv<UInt32>>.self)
    let sk = unbox(skPtr, as: SecretKey<Bfv<UInt32>>.self)
    do {
        let evalKey = try ctx.client.generateEvaluationKey(using: sk)
        return box(evalKey)
    } catch {
        print("[pnns32] gen_eval_key error: \(error)")
        return nil
    }
}

@_cdecl("pnns32_encrypt")
public func pnns32Encrypt(
    ctxPtr: UnsafeRawPointer,
    skPtr: UnsafeRawPointer,
    vector: UnsafePointer<Float>,
    dim: Int32) -> UnsafeMutableRawPointer?
{
    let ctx = unbox(ctxPtr, as: PnnsCtx<Bfv<UInt32>>.self)
    let sk = unbox(skPtr, as: SecretKey<Bfv<UInt32>>.self)
    do {
        let floats = Array(UnsafeBufferPointer(start: vector, count: Int(dim)))
        let vectors = Array2d(data: floats, rowCount: 1, columnCount: Int(dim))
        let query = try ctx.client.generateQuery(for: vectors, using: sk)
        return box(query)
    } catch {
        print("[pnns32] encrypt error: \(error)")
        return nil
    }
}

@_cdecl("pnns32_create_server")
public func pnns32CreateServer(
    ctxPtr: UnsafeRawPointer,
    dbVectors: UnsafePointer<Float>,
    dbRows: Int32,
    vectorDim: Int32) -> UnsafeMutableRawPointer?
{
    let ctx = unbox(ctxPtr, as: PnnsCtx<Bfv<UInt32>>.self)
    let count = Int(dbRows) * Int(vectorDim)
    let floats = Array(UnsafeBufferPointer(start: dbVectors, count: count))

    let rows = (0..<Int(dbRows)).map { i in
        let start = i * Int(vectorDim)
        let end = start + Int(vectorDim)
        return DatabaseRow(
            entryId: UInt64(i),
            entryMetadata: [],
            vector: Array(floats[start..<end]))
    }
    let database = Database(rows: rows)
    let config = ctx.serverConfig

    guard let server: PTServer<Bfv<UInt32>> = runBlocking({
        let processed = try await database.process(config: config)
        return try PTServer(database: processed)
    }) else { return nil }
    return box(server)
}

@_cdecl("pnns32_compute")
public func pnns32Compute(
    serverPtr: UnsafeRawPointer,
    queryPtr: UnsafeRawPointer,
    evalKeyPtr: UnsafeRawPointer,
    modSwitchDown: Int32) -> UnsafeMutableRawPointer?
{
    let server = unbox(serverPtr, as: PTServer<Bfv<UInt32>>.self)
    let query = unbox(queryPtr, as: Query<Bfv<UInt32>>.self)
    let evalKey = unbox(evalKeyPtr, as: EvaluationKey<Bfv<UInt32>>.self)
    let mod = modSwitchDown != 0

    guard let response: Response<Bfv<UInt32>> = runBlocking({
        try await server.computeResponse(to: query, using: evalKey, modSwitchDown: mod)
    }) else { return nil }
    return box(response)
}

@_cdecl("pnns32_decrypt")
public func pnns32Decrypt(
    ctxPtr: UnsafeRawPointer,
    skPtr: UnsafeRawPointer,
    responsePtr: UnsafeRawPointer,
    outDistances: UnsafeMutablePointer<Float>,
    outEntryIds: UnsafeMutablePointer<UInt64>,
    outLen: Int32) -> Int32
{
    let ctx = unbox(ctxPtr, as: PnnsCtx<Bfv<UInt32>>.self)
    let sk = unbox(skPtr, as: SecretKey<Bfv<UInt32>>.self)
    let response = unbox(responsePtr, as: Response<Bfv<UInt32>>.self)
    do {
        let result = try ctx.client.decrypt(response: response, using: sk)
        let shape = result.distances.shape
        let totalEntries = shape.rowCount
        guard totalEntries <= Int(outLen) else {
            print("[pnns32] decrypt: buffer too small (\(outLen) < \(totalEntries))")
            return -1
        }
        for r in 0..<shape.rowCount {
            for c in 0..<shape.columnCount {
                outDistances[r * shape.columnCount + c] = result.distances[r, c]
            }
        }
        for i in 0..<min(result.entryIds.count, Int(outLen)) {
            outEntryIds[i] = result.entryIds[i]
        }
        return Int32(totalEntries)
    } catch {
        print("[pnns32] decrypt error: \(error)")
        return -1
    }
}

@_cdecl("pnns32_save_db")
public func pnns32SaveDb(
    serverPtr: UnsafeRawPointer,
    path: UnsafePointer<CChar>) -> Int32
{
    let server = unbox(serverPtr, as: PTServer<Bfv<UInt32>>.self)
    let filePath = String(cString: path)
    do {
        let serialized = try server.database.serialize()
        let proto = try serialized.proto()
        try proto.save(to: filePath)
        return 0
    } catch {
        print("[pnns32] save_db error: \(error)")
        return -1
    }
}

@_cdecl("pnns32_load_server")
public func pnns32LoadServer(
    path: UnsafePointer<CChar>) -> UnsafeMutableRawPointer?
{
    let filePath = String(cString: path)
    do {
        let proto = try Apple_SwiftHomomorphicEncryption_Pnns_V1_SerializedProcessedDatabase(from: filePath)
        let serialized: SerializedProcessedDatabase<Bfv<UInt32>> = try proto.native()
        let processed = try ProcessedDatabase<Bfv<UInt32>>(from: serialized)
        let server = try PTServer(database: processed)
        return box(server)
    } catch {
        print("[pnns32] load_server error: \(error)")
        return nil
    }
}

@_cdecl("pnns32_load_server_with_config")
public func pnns32LoadServerWithConfig(
    path: UnsafePointer<CChar>,
    outVectorDim: UnsafeMutablePointer<Int32>,
    outDbRows: UnsafeMutablePointer<Int32>) -> UnsafeMutableRawPointer?
{
    let filePath = String(cString: path)
    do {
        let proto = try Apple_SwiftHomomorphicEncryption_Pnns_V1_SerializedProcessedDatabase(from: filePath)
        let serialized: SerializedProcessedDatabase<Bfv<UInt32>> = try proto.native()
        let processed = try ProcessedDatabase<Bfv<UInt32>>(from: serialized)
        let server = try PTServer(database: processed)
        outVectorDim.pointee = Int32(server.config.vectorDimension)
        outDbRows.pointee = Int32(server.database.entryIds.count)
        return box(server)
    } catch {
        print("[pnns32] load_server_with_config error: \(error)")
        return nil
    }
}

// MARK: - Bfv<UInt32> Fast Encoder

@_cdecl("pnns32_create_fast_encoder")
public func pnns32CreateFastEncoder(
    ctxPtr: UnsafeRawPointer,
    dbRows: Int32) -> UnsafeMutableRawPointer?
{
    let ctx = unbox(ctxPtr, as: PnnsCtx<Bfv<UInt32>>.self)
    do {
        let encoder = try FastPlaintextEncoder<Bfv<UInt32>>(
            config: ctx.serverConfig,
            context: ctx.context,
            rowCount: Int(dbRows))
        return box(encoder)
    } catch {
        print("[pnns32] create_fast_encoder error: \(error)")
        return nil
    }
}

@_cdecl("pnns32_quantize_db")
public func pnns32QuantizeDb(
    ctxPtr: UnsafeRawPointer,
    dbVectors: UnsafePointer<Float>,
    dbRows: Int32,
    vectorDim: Int32,
    outBytes: UnsafeMutablePointer<UInt8>,
    outBytesCapacity: Int,
    outEntryIds: UnsafeMutablePointer<UInt64>) -> Int
{
    let ctx = unbox(ctxPtr, as: PnnsCtx<Bfv<UInt32>>.self)
    let count = Int(dbRows) * Int(vectorDim)
    let floats = Array(UnsafeBufferPointer(start: dbVectors, count: count))

    let rows = (0..<Int(dbRows)).map { i in
        let start = i * Int(vectorDim)
        let end = start + Int(vectorDim)
        return DatabaseRow(
            entryId: UInt64(i),
            entryMetadata: [],
            vector: Array(floats[start..<end]))
    }
    let database = Database(rows: rows)
    let quantized = QuantizedDatabase<Bfv<UInt32>>(database: database, config: ctx.serverConfig)
    let bytes = quantized.serializeVectors()

    guard bytes.count <= outBytesCapacity else {
        print("[pnns32] quantize_db: buffer too small (\(outBytesCapacity) < \(bytes.count))")
        return -1
    }
    bytes.withUnsafeBufferPointer { buf in
        if let base = buf.baseAddress {
            outBytes.update(from: base, count: buf.count)
        }
    }
    for i in 0..<Int(dbRows) {
        outEntryIds[i] = quantized.entryIds[i]
    }
    return bytes.count
}

@_cdecl("pnns32_fast_encode_server")
public func pnns32FastEncodeServer(
    ctxPtr: UnsafeRawPointer,
    encoderPtr: UnsafeRawPointer,
    quantizedBytes: UnsafePointer<UInt8>,
    quantizedLen: Int,
    entryIds: UnsafePointer<UInt64>,
    dbRows: Int32,
    vectorDim: Int32) -> UnsafeMutableRawPointer?
{
    let ctx = unbox(ctxPtr, as: PnnsCtx<Bfv<UInt32>>.self)
    let encoder = unbox(encoderPtr, as: FastPlaintextEncoder<Bfv<UInt32>>.self)
    do {
        let byteArray = Array(UnsafeBufferPointer(start: quantizedBytes, count: quantizedLen))
        let ids = Array(UnsafeBufferPointer(start: entryIds, count: Int(dbRows)))
        let quantizedDB = QuantizedDatabase<Bfv<UInt32>>.deserializeVectors(
            from: byteArray,
            rowCount: Int(dbRows),
            vectorDimension: Int(vectorDim),
            entryIds: ids)
        let processed = try encoder.encodeDatabase(quantizedDB, context: ctx.context)
        let server = try PTServer(database: processed)
        return box(server)
    } catch {
        print("[pnns32] fast_encode_server error: \(error)")
        return nil
    }
}

// MARK: - Wire serialization

//
// Convention: every serialize function returns the number of bytes. Calling
// with `outBytes == nil` or `outCap == 0` is a size-probe — returns the
// required capacity without writing anything. Calling with a sufficient
// buffer writes the bytes and returns the write count. Insufficient capacity
// or internal failure returns -1 (message logged to stderr).
//
// Deserialize functions take the raw bytes plus whatever context handle is
// needed to reconstruct the native object and return an opaque handle (or
// NULL on error) that the caller must later free via `pnns_free_handle`.

private func writeBytes(
    _ data: Data,
    out: UnsafeMutablePointer<UInt8>?,
    cap: Int,
    tag: String) -> Int
{
    guard let out, cap > 0 else {
        return data.count
    }
    guard data.count <= cap else {
        print("[\(tag)] serialize: buffer too small (\(cap) < \(data.count))")
        return -1
    }
    data.withUnsafeBytes { raw in
        if let base = raw.baseAddress {
            out.update(from: base.assumingMemoryBound(to: UInt8.self), count: data.count)
        }
    }
    return data.count
}

// ---- SecretKey ------------------------------------------------------------

private func skSerialize<Scheme: HeScheme>(
    _: Scheme.Type,
    skPtr: UnsafeRawPointer,
    outBytes: UnsafeMutablePointer<UInt8>?,
    outCap: Int,
    tag: String) -> Int
{
    let sk = unbox(skPtr, as: SecretKey<Scheme>.self)
    do {
        let proto = sk.serialize().proto()
        let data = try proto.serializedData()
        return writeBytes(data, out: outBytes, cap: outCap, tag: tag)
    } catch {
        print("[\(tag)] sk_serialize error: \(error)")
        return -1
    }
}

private func skDeserialize<Scheme: HeScheme>(
    _: Scheme.Type,
    ctxPtr: UnsafeRawPointer,
    bytes: UnsafePointer<UInt8>,
    len: Int,
    tag: String) -> UnsafeMutableRawPointer?
{
    let ctx = unbox(ctxPtr, as: PnnsCtx<Scheme>.self)
    do {
        let data = Data(bytes: UnsafeRawPointer(bytes), count: len)
        let proto = try Apple_SwiftHomomorphicEncryption_V1_SerializedSecretKey(serializedBytes: data)
        let serialized = try proto.native()
        let sk = try SecretKey<Scheme>(deserialize: serialized, context: ctx.context)
        return box(sk)
    } catch {
        print("[\(tag)] sk_deserialize error: \(error)")
        return nil
    }
}

@_cdecl("pnns64_sk_serialize")
public func pnns64SkSerialize(
    skPtr: UnsafeRawPointer,
    outBytes: UnsafeMutablePointer<UInt8>?,
    outCap: Int) -> Int
{
    skSerialize(Bfv<UInt64>.self, skPtr: skPtr, outBytes: outBytes, outCap: outCap, tag: "pnns64")
}

@_cdecl("pnns64_sk_deserialize")
public func pnns64SkDeserialize(
    ctxPtr: UnsafeRawPointer,
    bytes: UnsafePointer<UInt8>,
    len: Int) -> UnsafeMutableRawPointer?
{
    skDeserialize(Bfv<UInt64>.self, ctxPtr: ctxPtr, bytes: bytes, len: len, tag: "pnns64")
}

@_cdecl("pnns32_sk_serialize")
public func pnns32SkSerialize(
    skPtr: UnsafeRawPointer,
    outBytes: UnsafeMutablePointer<UInt8>?,
    outCap: Int) -> Int
{
    skSerialize(Bfv<UInt32>.self, skPtr: skPtr, outBytes: outBytes, outCap: outCap, tag: "pnns32")
}

@_cdecl("pnns32_sk_deserialize")
public func pnns32SkDeserialize(
    ctxPtr: UnsafeRawPointer,
    bytes: UnsafePointer<UInt8>,
    len: Int) -> UnsafeMutableRawPointer?
{
    skDeserialize(Bfv<UInt32>.self, ctxPtr: ctxPtr, bytes: bytes, len: len, tag: "pnns32")
}

// ---- EvaluationKey --------------------------------------------------------

private func evalKeySerialize<Scheme: HeScheme>(
    _: Scheme.Type,
    evalKeyPtr: UnsafeRawPointer,
    outBytes: UnsafeMutablePointer<UInt8>?,
    outCap: Int,
    tag: String) -> Int
{
    let ek = unbox(evalKeyPtr, as: EvaluationKey<Scheme>.self)
    do {
        let proto = ek.serialize().proto()
        let data = try proto.serializedData()
        return writeBytes(data, out: outBytes, cap: outCap, tag: tag)
    } catch {
        print("[\(tag)] eval_key_serialize error: \(error)")
        return -1
    }
}

private func evalKeyDeserialize<Scheme: HeScheme>(
    _: Scheme.Type,
    ctxPtr: UnsafeRawPointer,
    bytes: UnsafePointer<UInt8>,
    len: Int,
    tag: String) -> UnsafeMutableRawPointer?
{
    let ctx = unbox(ctxPtr, as: PnnsCtx<Scheme>.self)
    do {
        let data = Data(bytes: UnsafeRawPointer(bytes), count: len)
        let proto = try Apple_SwiftHomomorphicEncryption_V1_SerializedEvaluationKey(serializedBytes: data)
        let ek = try proto.native(context: ctx.context) as EvaluationKey<Scheme>
        return box(ek)
    } catch {
        print("[\(tag)] eval_key_deserialize error: \(error)")
        return nil
    }
}

@_cdecl("pnns64_eval_key_serialize")
public func pnns64EvalKeySerialize(
    evalKeyPtr: UnsafeRawPointer,
    outBytes: UnsafeMutablePointer<UInt8>?,
    outCap: Int) -> Int
{
    evalKeySerialize(Bfv<UInt64>.self, evalKeyPtr: evalKeyPtr,
                     outBytes: outBytes, outCap: outCap, tag: "pnns64")
}

@_cdecl("pnns64_eval_key_deserialize")
public func pnns64EvalKeyDeserialize(
    ctxPtr: UnsafeRawPointer,
    bytes: UnsafePointer<UInt8>,
    len: Int) -> UnsafeMutableRawPointer?
{
    evalKeyDeserialize(Bfv<UInt64>.self, ctxPtr: ctxPtr, bytes: bytes, len: len, tag: "pnns64")
}

@_cdecl("pnns32_eval_key_serialize")
public func pnns32EvalKeySerialize(
    evalKeyPtr: UnsafeRawPointer,
    outBytes: UnsafeMutablePointer<UInt8>?,
    outCap: Int) -> Int
{
    evalKeySerialize(Bfv<UInt32>.self, evalKeyPtr: evalKeyPtr,
                     outBytes: outBytes, outCap: outCap, tag: "pnns32")
}

@_cdecl("pnns32_eval_key_deserialize")
public func pnns32EvalKeyDeserialize(
    ctxPtr: UnsafeRawPointer,
    bytes: UnsafePointer<UInt8>,
    len: Int) -> UnsafeMutableRawPointer?
{
    evalKeyDeserialize(Bfv<UInt32>.self, ctxPtr: ctxPtr, bytes: bytes, len: len, tag: "pnns32")
}

// ---- Query ----------------------------------------------------------------
// Query is `repeated SerializedCiphertextMatrix`. The PNNSShardResponse proto
// already has that shape, so we reuse it as a convenient envelope with empty
// entryIds / entryMetadatas.

private func querySerialize<Scheme: HeScheme>(
    _: Scheme.Type,
    queryPtr: UnsafeRawPointer,
    outBytes: UnsafeMutablePointer<UInt8>?,
    outCap: Int,
    tag: String) -> Int
{
    let query = unbox(queryPtr, as: Query<Scheme>.self)
    do {
        let matrices = try query.proto()
        var env = Apple_SwiftHomomorphicEncryption_Api_Pnns_V1_PNNSShardResponse()
        env.reply = matrices
        let data = try env.serializedData()
        return writeBytes(data, out: outBytes, cap: outCap, tag: tag)
    } catch {
        print("[\(tag)] query_serialize error: \(error)")
        return -1
    }
}

private func queryDeserialize<Scheme: HeScheme>(
    _: Scheme.Type,
    ctxPtr: UnsafeRawPointer,
    bytes: UnsafePointer<UInt8>,
    len: Int,
    tag: String) -> UnsafeMutableRawPointer?
{
    let ctx = unbox(ctxPtr, as: PnnsCtx<Scheme>.self)
    do {
        let data = Data(bytes: UnsafeRawPointer(bytes), count: len)
        let env = try Apple_SwiftHomomorphicEncryption_Api_Pnns_V1_PNNSShardResponse(serializedBytes: data)
        let query: Query<Scheme> = try env.reply.native(context: ctx.context)
        return box(query)
    } catch {
        print("[\(tag)] query_deserialize error: \(error)")
        return nil
    }
}

@_cdecl("pnns64_query_serialize")
public func pnns64QuerySerialize(
    queryPtr: UnsafeRawPointer,
    outBytes: UnsafeMutablePointer<UInt8>?,
    outCap: Int) -> Int
{
    querySerialize(Bfv<UInt64>.self, queryPtr: queryPtr,
                   outBytes: outBytes, outCap: outCap, tag: "pnns64")
}

@_cdecl("pnns64_query_deserialize")
public func pnns64QueryDeserialize(
    ctxPtr: UnsafeRawPointer,
    bytes: UnsafePointer<UInt8>,
    len: Int) -> UnsafeMutableRawPointer?
{
    queryDeserialize(Bfv<UInt64>.self, ctxPtr: ctxPtr, bytes: bytes, len: len, tag: "pnns64")
}

@_cdecl("pnns32_query_serialize")
public func pnns32QuerySerialize(
    queryPtr: UnsafeRawPointer,
    outBytes: UnsafeMutablePointer<UInt8>?,
    outCap: Int) -> Int
{
    querySerialize(Bfv<UInt32>.self, queryPtr: queryPtr,
                   outBytes: outBytes, outCap: outCap, tag: "pnns32")
}

@_cdecl("pnns32_query_deserialize")
public func pnns32QueryDeserialize(
    ctxPtr: UnsafeRawPointer,
    bytes: UnsafePointer<UInt8>,
    len: Int) -> UnsafeMutableRawPointer?
{
    queryDeserialize(Bfv<UInt32>.self, ctxPtr: ctxPtr, bytes: bytes, len: len, tag: "pnns32")
}

// ---- Response -------------------------------------------------------------

private func responseSerialize<Scheme: HeScheme>(
    _: Scheme.Type,
    responsePtr: UnsafeRawPointer,
    outBytes: UnsafeMutablePointer<UInt8>?,
    outCap: Int,
    tag: String) -> Int
{
    let response = unbox(responsePtr, as: Response<Scheme>.self)
    do {
        let proto = try response.proto()
        let data = try proto.serializedData()
        return writeBytes(data, out: outBytes, cap: outCap, tag: tag)
    } catch {
        print("[\(tag)] response_serialize error: \(error)")
        return -1
    }
}

private func responseDeserialize<Scheme: HeScheme>(
    _: Scheme.Type,
    ctxPtr: UnsafeRawPointer,
    bytes: UnsafePointer<UInt8>,
    len: Int,
    moduliCount: Int32,
    tag: String) -> UnsafeMutableRawPointer?
{
    let ctx = unbox(ctxPtr, as: PnnsCtx<Scheme>.self)
    do {
        let data = Data(bytes: UnsafeRawPointer(bytes), count: len)
        let proto = try Apple_SwiftHomomorphicEncryption_Api_Pnns_V1_PNNSShardResponse(serializedBytes: data)
        // moduliCount: pass -1 to use all moduli (for un-modswitched responses).
        // Pass 0 to default to 1 (normal modSwitchDown=true responses).
        // Positive values are taken verbatim.
        let moduli: Int? = moduliCount < 0 ? nil : (moduliCount == 0 ? 1 : Int(moduliCount))
        let response = try proto.native(contexts: [ctx.context], moduliCount: moduli) as Response<Scheme>
        return box(response)
    } catch {
        print("[\(tag)] response_deserialize error: \(error)")
        return nil
    }
}

@_cdecl("pnns64_response_serialize")
public func pnns64ResponseSerialize(
    responsePtr: UnsafeRawPointer,
    outBytes: UnsafeMutablePointer<UInt8>?,
    outCap: Int) -> Int
{
    responseSerialize(Bfv<UInt64>.self, responsePtr: responsePtr,
                      outBytes: outBytes, outCap: outCap, tag: "pnns64")
}

@_cdecl("pnns64_response_deserialize")
public func pnns64ResponseDeserialize(
    ctxPtr: UnsafeRawPointer,
    bytes: UnsafePointer<UInt8>,
    len: Int,
    moduliCount: Int32) -> UnsafeMutableRawPointer?
{
    responseDeserialize(Bfv<UInt64>.self, ctxPtr: ctxPtr, bytes: bytes, len: len,
                        moduliCount: moduliCount, tag: "pnns64")
}

@_cdecl("pnns32_response_serialize")
public func pnns32ResponseSerialize(
    responsePtr: UnsafeRawPointer,
    outBytes: UnsafeMutablePointer<UInt8>?,
    outCap: Int) -> Int
{
    responseSerialize(Bfv<UInt32>.self, responsePtr: responsePtr,
                      outBytes: outBytes, outCap: outCap, tag: "pnns32")
}

@_cdecl("pnns32_response_deserialize")
public func pnns32ResponseDeserialize(
    ctxPtr: UnsafeRawPointer,
    bytes: UnsafePointer<UInt8>,
    len: Int,
    moduliCount: Int32) -> UnsafeMutableRawPointer?
{
    responseDeserialize(Bfv<UInt32>.self, ctxPtr: ctxPtr, bytes: bytes, len: len,
                        moduliCount: moduliCount, tag: "pnns32")
}

// MARK: - Ciphertext server (CT PNNS)

//
// Mirrors the plaintext `pnns{64,32}_*` server API with the `ctpnns{64,32}_*`
// prefix. Client API (keygen / encrypt / decrypt / gen_eval_key) is shared —
// only the server side differs. Save/load use a single-file proto format
// (SerializedEncryptedProcessedDatabase) to match the plaintext shape.

private func ctCreateServer<Scheme: HeScheme>(
    _: Scheme.Type,
    ctxPtr: UnsafeRawPointer,
    dbVectors: UnsafePointer<Float>,
    skPtr: UnsafeRawPointer,
    dbRows: Int32,
    vectorDim: Int32,
    tag: String) -> UnsafeMutableRawPointer?
{
    let ctx = unbox(ctxPtr, as: PnnsCtx<Scheme>.self)
    let sk = unbox(skPtr, as: SecretKey<Scheme>.self)
    let count = Int(dbRows) * Int(vectorDim)
    let floats = Array(UnsafeBufferPointer(start: dbVectors, count: count))

    let rows = (0..<Int(dbRows)).map { i in
        let start = i * Int(vectorDim)
        let end = start + Int(vectorDim)
        return DatabaseRow(
            entryId: UInt64(i),
            entryMetadata: [],
            vector: Array(floats[start..<end]))
    }
    let database = Database(rows: rows)
    do {
        let encrypted = try database.processAndEncrypt(
            config: ctx.serverConfig,
            secretKey: sk,
            contexts: [ctx.context])
        let server = CTServer(database: encrypted)
        return box(server)
    } catch {
        print("[\(tag)] ct_create_server error: \(error)")
        return nil
    }
}

private func ctCompute<Scheme: HeScheme>(
    _: Scheme.Type,
    serverPtr: UnsafeRawPointer,
    queryPtr: UnsafeRawPointer,
    evalKeyPtr: UnsafeRawPointer,
    tag _: String) -> UnsafeMutableRawPointer?
{
    let server = unbox(serverPtr, as: CTServer<Scheme>.self)
    let query = unbox(queryPtr, as: Query<Scheme>.self)
    let evalKey = unbox(evalKeyPtr, as: EvaluationKey<Scheme>.self)

    guard let response: Response<Scheme> = runBlocking({
        try await server.computeResponse(to: query, using: evalKey)
    }) else { return nil }
    return box(response)
}

private func ctSaveDb<Scheme: HeScheme>(
    _: Scheme.Type,
    serverPtr: UnsafeRawPointer,
    path: UnsafePointer<CChar>,
    tag: String) -> Int32
{
    let server = unbox(serverPtr, as: CTServer<Scheme>.self)
    let filePath = String(cString: path)
    do {
        let serialized = try server.database.serialize()
        let proto = try serialized.proto()
        try proto.save(to: filePath)
        return 0
    } catch {
        print("[\(tag)] ct_save_db error: \(error)")
        return -1
    }
}

private func ctLoadServer<Scheme: HeScheme>(
    _: Scheme.Type,
    path: UnsafePointer<CChar>,
    tag: String) -> (CTServer<Scheme>, EncryptedProcessedDatabase<Scheme>)?
{
    let filePath = String(cString: path)
    do {
        let proto = try Apple_SwiftHomomorphicEncryption_Pnns_V1_SerializedEncryptedProcessedDatabase(
            from: filePath)
        let serialized: SerializedEncryptedProcessedDatabase<Scheme> = try proto.native()
        let encrypted = try EncryptedProcessedDatabase<Scheme>(from: serialized)
        let server = CTServer(database: encrypted)
        return (server, encrypted)
    } catch {
        print("[\(tag)] ct_load_server error: \(error)")
        return nil
    }
}

@_cdecl("ctpnns64_create_server")
public func ctpnns64CreateServer(
    ctxPtr: UnsafeRawPointer,
    dbVectors: UnsafePointer<Float>,
    skPtr: UnsafeRawPointer,
    dbRows: Int32,
    vectorDim: Int32) -> UnsafeMutableRawPointer?
{
    ctCreateServer(Bfv<UInt64>.self, ctxPtr: ctxPtr, dbVectors: dbVectors,
                   skPtr: skPtr, dbRows: dbRows, vectorDim: vectorDim, tag: "ctpnns64")
}

@_cdecl("ctpnns64_compute")
public func ctpnns64Compute(
    serverPtr: UnsafeRawPointer,
    queryPtr: UnsafeRawPointer,
    evalKeyPtr: UnsafeRawPointer) -> UnsafeMutableRawPointer?
{
    ctCompute(Bfv<UInt64>.self, serverPtr: serverPtr, queryPtr: queryPtr,
              evalKeyPtr: evalKeyPtr, tag: "ctpnns64")
}

@_cdecl("ctpnns64_save_db")
public func ctpnns64SaveDb(
    serverPtr: UnsafeRawPointer,
    path: UnsafePointer<CChar>) -> Int32
{
    ctSaveDb(Bfv<UInt64>.self, serverPtr: serverPtr, path: path, tag: "ctpnns64")
}

@_cdecl("ctpnns64_load_server")
public func ctpnns64LoadServer(
    path: UnsafePointer<CChar>) -> UnsafeMutableRawPointer?
{
    guard let (server, _) = ctLoadServer(Bfv<UInt64>.self, path: path, tag: "ctpnns64")
    else { return nil }
    return box(server)
}

@_cdecl("ctpnns64_load_server_with_config")
public func ctpnns64LoadServerWithConfig(
    path: UnsafePointer<CChar>,
    outVectorDim: UnsafeMutablePointer<Int32>,
    outDbRows: UnsafeMutablePointer<Int32>) -> UnsafeMutableRawPointer?
{
    guard let (server, db) = ctLoadServer(Bfv<UInt64>.self, path: path, tag: "ctpnns64")
    else { return nil }
    outVectorDim.pointee = Int32(db.serverConfig.vectorDimension)
    outDbRows.pointee = Int32(db.entryIds.count)
    return box(server)
}

@_cdecl("ctpnns32_create_server")
public func ctpnns32CreateServer(
    ctxPtr: UnsafeRawPointer,
    dbVectors: UnsafePointer<Float>,
    skPtr: UnsafeRawPointer,
    dbRows: Int32,
    vectorDim: Int32) -> UnsafeMutableRawPointer?
{
    ctCreateServer(Bfv<UInt32>.self, ctxPtr: ctxPtr, dbVectors: dbVectors,
                   skPtr: skPtr, dbRows: dbRows, vectorDim: vectorDim, tag: "ctpnns32")
}

@_cdecl("ctpnns32_compute")
public func ctpnns32Compute(
    serverPtr: UnsafeRawPointer,
    queryPtr: UnsafeRawPointer,
    evalKeyPtr: UnsafeRawPointer) -> UnsafeMutableRawPointer?
{
    ctCompute(Bfv<UInt32>.self, serverPtr: serverPtr, queryPtr: queryPtr,
              evalKeyPtr: evalKeyPtr, tag: "ctpnns32")
}

@_cdecl("ctpnns32_save_db")
public func ctpnns32SaveDb(
    serverPtr: UnsafeRawPointer,
    path: UnsafePointer<CChar>) -> Int32
{
    ctSaveDb(Bfv<UInt32>.self, serverPtr: serverPtr, path: path, tag: "ctpnns32")
}

@_cdecl("ctpnns32_load_server")
public func ctpnns32LoadServer(
    path: UnsafePointer<CChar>) -> UnsafeMutableRawPointer?
{
    guard let (server, _) = ctLoadServer(Bfv<UInt32>.self, path: path, tag: "ctpnns32")
    else { return nil }
    return box(server)
}

@_cdecl("ctpnns32_load_server_with_config")
public func ctpnns32LoadServerWithConfig(
    path: UnsafePointer<CChar>,
    outVectorDim: UnsafeMutablePointer<Int32>,
    outDbRows: UnsafeMutablePointer<Int32>) -> UnsafeMutableRawPointer?
{
    guard let (server, db) = ctLoadServer(Bfv<UInt32>.self, path: path, tag: "ctpnns32")
    else { return nil }
    outVectorDim.pointee = Int32(db.serverConfig.vectorDimension)
    outDbRows.pointee = Int32(db.entryIds.count)
    return box(server)
}

// MARK: - Response aggregation

/// Homomorphically aggregate an array of Response<Bfv<UInt64>> handles into a
/// single Response handle. `responses` is a C array of opaque response pointers
/// of length `count`. Returns NULL on error (empty array, dimension mismatch).
/// The returned handle must be freed with `pnns_free_handle`. Input handles
/// are not consumed.
@_cdecl("pnns64_response_aggregate")
public func pnns64ResponseAggregate(
    responses: UnsafePointer<UnsafeRawPointer?>,
    count: Int) -> UnsafeMutableRawPointer?
{
    guard count > 0 else {
        print("[pnns64] response_aggregate: empty input")
        return nil
    }
    var collected: [Response<Bfv<UInt64>>] = []
    collected.reserveCapacity(count)
    for i in 0..<count {
        guard let p = responses[i] else {
            print("[pnns64] response_aggregate: NULL at index \(i)")
            return nil
        }
        collected.append(unbox(p, as: Response<Bfv<UInt64>>.self))
    }
    do {
        let aggregated = try Response<Bfv<UInt64>>.aggregate(collected)
        return box(aggregated)
    } catch {
        print("[pnns64] response_aggregate error: \(error)")
        return nil
    }
}

/// See `pnns64_response_aggregate`.
@_cdecl("pnns32_response_aggregate")
public func pnns32ResponseAggregate(
    responses: UnsafePointer<UnsafeRawPointer?>,
    count: Int) -> UnsafeMutableRawPointer?
{
    guard count > 0 else {
        print("[pnns32] response_aggregate: empty input")
        return nil
    }
    var collected: [Response<Bfv<UInt32>>] = []
    collected.reserveCapacity(count)
    for i in 0..<count {
        guard let p = responses[i] else {
            print("[pnns32] response_aggregate: NULL at index \(i)")
            return nil
        }
        collected.append(unbox(p, as: Response<Bfv<UInt32>>.self))
    }
    do {
        let aggregated = try Response<Bfv<UInt32>>.aggregate(collected)
        return box(aggregated)
    } catch {
        print("[pnns32] response_aggregate error: \(error)")
        return nil
    }
}

// MARK: - Memory management

@_cdecl("pnns_free_handle")
public func pnnsFreeHandle(ptr: UnsafeMutableRawPointer) {
    release(ptr)
}

// swiftlint:enable function_parameter_count missing_docs
