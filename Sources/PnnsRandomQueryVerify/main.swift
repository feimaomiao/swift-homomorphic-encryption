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

// Verification script: does the server happily accept a query (and
// evaluation key) constructed from uniform random bits of the correct shape?
//
// Under RLWE, a real query/eval key is computationally indistinguishable
// from uniform. We rebuild the polynomials of a real query (and, in stage 2,
// of the evaluation key) with random coefficients in [0, q_i) per RNS modulus,
// feed them to Server.computeResponse, and observe whether the server runs
// to completion. Decrypted output is expected to be garbage; we just check
// the server path doesn't error.

import Foundation
import HomomorphicEncryption
import ModularArithmetic
import PrivateNearestNeighborSearch

typealias Scheme = Bfv<UInt64>

@main
enum PnnsRandomQueryVerify {
    // MARK: - Subtypes

    struct Fixture {
        let client: Client<Scheme>
        let server: Server<Scheme>
        let secretKey: SecretKey<Scheme>
        let realQuery: Query<Scheme>
        let realEvalKey: EvaluationKey<Scheme>
    }

    // Padded coefficient u satisfies u mod q_i == c, where c is the original
    // reduced coefficient. Since u may exceed q_i, it can't live inside a
    // PolyRq (which asserts coeff < q_i). We keep padded values in plain
    // [UInt64] arrays, shaped identically to the real PolyRq.data layout.
    // No randomization; this is a reversible transform. Server-side unpad is
    // `u % q_i` per RNS row.

    struct PaddedPoly {
        let rowCount: Int
        let columnCount: Int
        let data: [UInt64] // row-major: rowCount * columnCount entries
    }

    struct PaddedCiphertext {
        let polys: [PaddedPoly]
    }

    struct PaddedCiphertextMatrix {
        let ciphertexts: [PaddedCiphertext]
    }

    struct PaddedQuery {
        let matrices: [PaddedCiphertextMatrix]
        var totalCoefficients: Int {
            matrices.flatMap(\.ciphertexts).flatMap(\.polys).reduce(0) { $0 + $1.data.count }
        }
    }

    struct PaddedKSK {
        let ciphertexts: [PaddedCiphertext]
    }

    struct PaddedEvalKey {
        let galoisKeys: [Int: PaddedKSK]
        let relinKey: PaddedKSK?
        var totalCoefficients: Int {
            let gc = galoisKeys.values.flatMap(\.ciphertexts).flatMap(\.polys)
                .reduce(0) { $0 + $1.data.count }
            let rc = relinKey?.ciphertexts.flatMap(\.polys).reduce(0) { $0 + $1.data.count } ?? 0
            return gc + rc
        }
    }

    // MARK: - Entry point

    static func main() async throws {
        let fx = try await buildFixture()

        try await stageSanity(fx)
        let randomQuery = try await stageRandomQuery(fx)
        try await stageRandomBoth(fx, randomQuery: randomQuery)
        try await stagePadded(fx)
    }

    // MARK: - Setup

    static func buildFixture() async throws -> Fixture {
        // Config mirrors _TestUtilities/PnnsUtilities/ClientTests.swift:244-249.
        // securityLevel: .unchecked is required because degree=64 is below
        // production HE security thresholds — do NOT copy this config elsewhere.
        let degree = 64
        let plaintextModuliCount = 2
        let plaintextModuli = try UInt64.generatePrimes(
            significantBitCounts: Array(repeating: 10, count: plaintextModuliCount),
            preferringSmall: true,
            nttDegree: degree)
        let coefficientModuli = try UInt64.generatePrimes(
            significantBitCounts: Array(repeating: UInt64.bitWidth - 4, count: 3),
            preferringSmall: false,
            nttDegree: degree)
        let encryptionParameters = try EncryptionParameters<UInt64>(
            polyDegree: degree,
            plaintextModulus: plaintextModuli[0],
            coefficientModuli: coefficientModuli,
            errorStdDev: .stdDev32,
            securityLevel: .unchecked)

        let dimensions = try MatrixDimensions(rowCount: degree, columnCount: 16)
        let vectorDimension = dimensions.columnCount
        let scalingFactor = ClientConfig<Scheme>.maxScalingFactor(
            distanceMetric: .cosineSimilarity,
            vectorDimension: vectorDimension,
            plaintextModuli: plaintextModuli)
        let evaluationKeyConfig = try MatrixMultiplication.evaluationKeyConfig(
            plaintextMatrixDimensions: dimensions,
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
            databasePacking: .diagonal(babyStepGiantStep: BabyStepGiantStep(vectorDimension: vectorDimension)))

        let rows = (0..<dimensions.rowCount).map { i in
            DatabaseRow(
                entryId: UInt64(i),
                entryMetadata: [],
                vector: (0..<dimensions.columnCount).map { j in
                    Float(j + i) * (i.isMultiple(of: 2) ? 1 : -1)
                })
        }
        let database = Database(rows: rows)
        let processed: ProcessedDatabase<Scheme> = try await database.process(config: serverConfig)
        let client = try Client(config: clientConfig, contexts: processed.contexts)
        let server = try Server<Scheme>(database: processed)

        let queryVectors = Array2d<Float>(data: [database.rows[0].vector])
        let secretKey = try client.generateSecretKey()
        let realQuery = try client.generateQuery(for: queryVectors, using: secretKey)
        let realEvalKey = try client.generateEvaluationKey(using: secretKey)

        return Fixture(client: client, server: server, secretKey: secretKey,
                       realQuery: realQuery, realEvalKey: realEvalKey)
    }

    // MARK: - Stages

    static func stageSanity(_ fx: Fixture) async throws {
        print("=== Sanity: real query, real eval key ===")
        try await runAndReport(fx, query: fx.realQuery, evalKey: fx.realEvalKey)
    }

    static func stageRandomQuery(_ fx: Fixture) async throws -> Query<Scheme> {
        print("\n=== Stage 1: random query, real eval key ===")
        let randomQuery = try randomizeQuery(fx.realQuery)
        try await runAndReport(fx, query: randomQuery, evalKey: fx.realEvalKey)
        return randomQuery
    }

    static func stageRandomBoth(_ fx: Fixture, randomQuery: Query<Scheme>) async throws {
        print("\n=== Stage 2: random query, random eval key ===")
        let randomEvalKey = try randomizeEvalKey(fx.realEvalKey)
        try await runAndReport(fx, query: randomQuery, evalKey: randomEvalKey)
    }

    // Stage 3: pad each coefficient c -> u in [0, 2^64) with u % q_i == c at
    // the client; server unpads (u % q_i) and rebuilds PolyRq/Ciphertext/....
    // Expected: result matches the sanity run exactly.
    static func stagePadded(_ fx: Fixture) async throws {
        print("\n=== Stage 3: pad at client, unpad at server, evaluate ===")
        let paddedQuery = extractPaddedQuery(fx.realQuery)
        let paddedEvalKey = extractPaddedEvalKey(fx.realEvalKey)
        let queryCount = paddedQuery.totalCoefficients
        let keyCount = paddedEvalKey.totalCoefficients
        print("  padded query UInt64 count: \(queryCount) (= \(queryCount * 8) bytes)")
        print("  padded eval key UInt64 count: \(keyCount) (= \(keyCount * 8) bytes)")
        verifyPadRoundTrip(realQuery: fx.realQuery, padded: paddedQuery)

        let reconstructedQuery = try reconstructQuery(paddedQuery, templateQuery: fx.realQuery)
        let reconstructedEvalKey = try reconstructEvalKey(paddedEvalKey, templateEvalKey: fx.realEvalKey)
        try await runAndReport(fx, query: reconstructedQuery, evalKey: reconstructedEvalKey)
    }

    static func verifyPadRoundTrip(realQuery: Query<Scheme>, padded: PaddedQuery) {
        let origCoeff = realQuery.ciphertextMatrices[0].ciphertexts[0].polys[0].data.data[0]
        let paddedCoeff = padded.matrices[0].ciphertexts[0].polys[0].data[0]
        let q0 = realQuery.ciphertextMatrices[0].ciphertexts[0].polys[0].context.moduli[0]
        let reduced = paddedCoeff % q0
        let match = reduced == origCoeff
        print("  pad round-trip check:")
        print("    original=\(origCoeff)")
        print("    padded=\(paddedCoeff)")
        print("    padded % q0=\(reduced), match=\(match)")
    }

    // MARK: - Server call + reporting

    static func runAndReport(_ fx: Fixture,
                             query: Query<Scheme>,
                             evalKey: EvaluationKey<Scheme>) async throws
    {
        do {
            let response = try await fx.server.computeResponse(to: query, using: evalKey)
            let matrixCount = response.ciphertextMatrices.count
            let entryCount = response.entryIds.count
            print("  server.computeResponse succeeded:")
            print("    \(matrixCount) ciphertext matrix/matrices, \(entryCount) entry ids")

            if let noise = try? response.noiseBudget(using: fx.secretKey, variableTime: true) {
                print("  response noise budget: \(noise)")
            }
            if let decrypted = try? fx.client.decrypt(response: response, using: fx.secretKey) {
                let preview = Array(decrypted.distances.data.prefix(8))
                print("  decrypted distances[0..<\(preview.count)]: \(preview)")
            } else {
                print("  decryption threw (expected when inputs are garbage)")
            }
        } catch {
            print("  server.computeResponse FAILED: \(error)")
        }
    }

    // MARK: - Randomization helpers

    static func randomizePoly<F: PolyFormat>(_ poly: PolyRq<UInt64, F>) -> PolyRq<UInt64, F> {
        let moduli = poly.context.moduli
        let rowCount = poly.data.rowCount
        let columnCount = poly.data.columnCount
        var flat = [UInt64](repeating: 0, count: rowCount * columnCount)
        for rns in 0..<rowCount {
            let q = moduli[rns]
            let base = rns * columnCount
            for c in 0..<columnCount {
                flat[base + c] = UInt64.random(in: 0..<q)
            }
        }
        let data = Array2d(data: flat, rowCount: rowCount, columnCount: columnCount)
        return PolyRq<UInt64, F>(context: poly.context, data: data)
    }

    static func randomizeCiphertext<F: PolyFormat>(
        _ ct: Ciphertext<Scheme, F>) throws -> Ciphertext<Scheme, F>
    {
        let newPolys = ct.polys.map(randomizePoly)
        return try Ciphertext(
            _context: ct.context,
            _polys: newPolys,
            _correctionFactor: ct.correctionFactor,
            _auxiliaryData: ct.auxiliaryData,
            _seed: [])
    }

    static func randomizeQuery(_ query: Query<Scheme>) throws -> Query<Scheme> {
        let mutated = try query.ciphertextMatrices.map { ctMatrix in
            let newCts = try ctMatrix.ciphertexts.map { try randomizeCiphertext($0) }
            return try CiphertextMatrix<Scheme, Coeff>(
                dimensions: ctMatrix.dimensions,
                packing: ctMatrix.packing,
                ciphertexts: newCts)
        }
        return Query(ciphertextMatrices: mutated)
    }

    static func randomizeKeySwitchKey(_ ksk: _KeySwitchKey<Scheme>) throws -> _KeySwitchKey<Scheme> {
        let newCts = try ksk._ciphertexts.map { try randomizeCiphertext($0) }
        return _KeySwitchKey<Scheme>(_context: ksk._context, _ciphertexts: newCts)
    }

    static func randomizeEvalKey(_ key: EvaluationKey<Scheme>) throws -> EvaluationKey<Scheme> {
        let newGalois: _GaloisKey<Scheme>?
        if let galois = key._galoisKey {
            var newKeys: [Int: _KeySwitchKey<Scheme>] = [:]
            for (element, ksk) in galois._keys {
                newKeys[element] = try randomizeKeySwitchKey(ksk)
            }
            newGalois = _GaloisKey<Scheme>(_keys: newKeys)
        } else {
            newGalois = nil
        }
        let newRelin: _RelinearizationKey<Scheme>?
        if let relin = key._relinearizationKey {
            let newKsk = try randomizeKeySwitchKey(relin._keySwitchKey)
            newRelin = _RelinearizationKey<Scheme>(_keySwitchKey: newKsk)
        } else {
            newRelin = nil
        }
        return EvaluationKey<Scheme>(_galoisKey: newGalois, _relinearizationKey: newRelin)
    }

    // MARK: - Wire-format padding (stage 3)

    // Pad: pick u uniformly in {c, c+q, c+2q, ..., c + kMax*q} where
    // kMax = floor((UInt64.max - c) / q). That is the largest arithmetic
    // progression with step q that fits in UInt64 starting at c.
    static func padCoefficient(_ c: UInt64, modulus q: UInt64) -> UInt64 {
        let kMax = (UInt64.max - c) / q
        let k = UInt64.random(in: 0...kMax)
        return c &+ k &* q
    }

    static func padPoly(_ poly: PolyRq<UInt64, some PolyFormat>) -> PaddedPoly {
        let moduli = poly.context.moduli
        let rowCount = poly.data.rowCount
        let columnCount = poly.data.columnCount
        var out = [UInt64](repeating: 0, count: rowCount * columnCount)
        for rns in 0..<rowCount {
            let q = moduli[rns]
            let base = rns * columnCount
            for c in 0..<columnCount {
                out[base + c] = padCoefficient(poly.data.data[base + c], modulus: q)
            }
        }
        return PaddedPoly(rowCount: rowCount, columnCount: columnCount, data: out)
    }

    static func padCiphertext(_ ct: Ciphertext<Scheme, some PolyFormat>) -> PaddedCiphertext {
        PaddedCiphertext(polys: ct.polys.map(padPoly))
    }

    static func extractPaddedQuery(_ query: Query<Scheme>) -> PaddedQuery {
        PaddedQuery(matrices: query.ciphertextMatrices.map { m in
            PaddedCiphertextMatrix(ciphertexts: m.ciphertexts.map(padCiphertext))
        })
    }

    static func extractPaddedEvalKey(_ key: EvaluationKey<Scheme>) -> PaddedEvalKey {
        var galois: [Int: PaddedKSK] = [:]
        if let g = key._galoisKey {
            for (elt, ksk) in g._keys {
                galois[elt] = PaddedKSK(ciphertexts: ksk._ciphertexts.map(padCiphertext))
            }
        }
        var relin: PaddedKSK?
        if let r = key._relinearizationKey {
            relin = PaddedKSK(ciphertexts: r._keySwitchKey._ciphertexts.map(padCiphertext))
        }
        return PaddedEvalKey(galoisKeys: galois, relinKey: relin)
    }

    /// Unpad (server side): reduce each coefficient mod q_i, rebuild PolyRq.
    /// The template poly supplies context (moduli) and format.
    static func unpadPoly<F: PolyFormat>(_ padded: PaddedPoly,
                                         template: PolyRq<UInt64, F>) -> PolyRq<UInt64, F>
    {
        let moduli = template.context.moduli
        let rowCount = padded.rowCount
        let columnCount = padded.columnCount
        precondition(rowCount == template.data.rowCount)
        precondition(columnCount == template.data.columnCount)
        var reduced = [UInt64](repeating: 0, count: rowCount * columnCount)
        for rns in 0..<rowCount {
            let q = moduli[rns]
            let base = rns * columnCount
            for c in 0..<columnCount {
                reduced[base + c] = padded.data[base + c] % q
            }
        }
        let data = Array2d(data: reduced, rowCount: rowCount, columnCount: columnCount)
        return PolyRq(context: template.context, data: data)
    }

    static func unpadCiphertext<F: PolyFormat>(_ padded: PaddedCiphertext,
                                               template: Ciphertext<Scheme, F>) throws -> Ciphertext<Scheme, F>
    {
        precondition(padded.polys.count == template.polys.count)
        let newPolys = zip(padded.polys, template.polys).map { unpadPoly($0, template: $1) }
        return try Ciphertext(
            _context: template.context,
            _polys: newPolys,
            _correctionFactor: template.correctionFactor,
            _auxiliaryData: template.auxiliaryData,
            _seed: [])
    }

    static func reconstructQuery(_ padded: PaddedQuery,
                                 templateQuery: Query<Scheme>) throws -> Query<Scheme>
    {
        precondition(padded.matrices.count == templateQuery.ciphertextMatrices.count)
        let newMatrices = try zip(padded.matrices, templateQuery.ciphertextMatrices).map { pad, tpl in
            precondition(pad.ciphertexts.count == tpl.ciphertexts.count)
            let newCts = try zip(pad.ciphertexts, tpl.ciphertexts).map { try unpadCiphertext($0, template: $1) }
            return try CiphertextMatrix<Scheme, Coeff>(
                dimensions: tpl.dimensions,
                packing: tpl.packing,
                ciphertexts: newCts)
        }
        return Query(ciphertextMatrices: newMatrices)
    }

    static func reconstructKSK(_ padded: PaddedKSK,
                               template: _KeySwitchKey<Scheme>) throws -> _KeySwitchKey<Scheme>
    {
        precondition(padded.ciphertexts.count == template._ciphertexts.count)
        let newCts = try zip(padded.ciphertexts, template._ciphertexts)
            .map { try unpadCiphertext($0, template: $1) }
        return _KeySwitchKey<Scheme>(_context: template._context, _ciphertexts: newCts)
    }

    static func reconstructEvalKey(_ padded: PaddedEvalKey,
                                   templateEvalKey: EvaluationKey<Scheme>) throws -> EvaluationKey<Scheme>
    {
        let newGalois: _GaloisKey<Scheme>?
        if let g = templateEvalKey._galoisKey {
            var newKeys: [Int: _KeySwitchKey<Scheme>] = [:]
            for (elt, tksk) in g._keys {
                guard let pksk = padded.galoisKeys[elt] else {
                    preconditionFailure("padded eval key missing galois element \(elt)")
                }
                newKeys[elt] = try reconstructKSK(pksk, template: tksk)
            }
            newGalois = _GaloisKey<Scheme>(_keys: newKeys)
        } else {
            newGalois = nil
        }
        let newRelin: _RelinearizationKey<Scheme>?
        if let r = templateEvalKey._relinearizationKey, let pr = padded.relinKey {
            let newKsk = try reconstructKSK(pr, template: r._keySwitchKey)
            newRelin = _RelinearizationKey<Scheme>(_keySwitchKey: newKsk)
        } else {
            newRelin = nil
        }
        return EvaluationKey<Scheme>(_galoisKey: newGalois, _relinearizationKey: newRelin)
    }
}
