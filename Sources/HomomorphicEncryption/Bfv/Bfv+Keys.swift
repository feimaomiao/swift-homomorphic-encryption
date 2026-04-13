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

public import ModularArithmetic

/// Precomputed digit decomposition for hoisted key switching.
///
/// Stores NTT'd digits of a polynomial for reuse across multiple Galois automorphisms
/// applied to the same ciphertext. This avoids redundant decomposition and forward NTT
/// computation per automorphism.
public struct KeySwitchDecomposition<T: ScalarType>: Sendable {
    /// Flat array of NTT'd digits.
    /// Indexed as `[rnsIndex * decomposeModuliCount * degree + decomposeIndex * degree + columnIndex]`.
    @usableFromInline let nttDigits: [T]
    /// Polynomial degree.
    @usableFromInline let degree: Int
    /// Number of decomposition moduli (= number of ciphertext moduli).
    @usableFromInline let decomposeModuliCount: Int
    /// Number of RNS moduli in extended basis (= decomposeModuliCount + 1).
    @usableFromInline let rnsModuliCount: Int

    @inlinable
    init(nttDigits: [T], degree: Int, decomposeModuliCount: Int, rnsModuliCount: Int) {
        self.nttDigits = nttDigits
        self.degree = degree
        self.decomposeModuliCount = decomposeModuliCount
        self.rnsModuliCount = rnsModuliCount
    }
}

extension Bfv {
    @inlinable
    // swiftlint:disable:next missing_docs attributes
    public static func generateSecretKey(context: Context) throws -> SecretKey<Bfv<T>> {
        var s = PolyRq<Scalar, Coeff>.zero(context: context.secretKeyContext)
        var rng = SystemRandomNumberGenerator()
        s.randomizeTernary(using: &rng)

        return try SecretKey(poly: s.forwardNtt())
    }

    @inlinable
    // swiftlint:disable:next missing_docs attributes
    public static func generateEvaluationKey(
        context: Context,
        config: EvaluationKeyConfig,
        using secretKey: borrowing SecretKey<Bfv<T>>) throws -> EvaluationKey<Bfv<T>>
    {
        guard context.supportsEvaluationKey else {
            throw HeError.unsupportedHeOperation()
        }
        var galoisKeys: [Int: Self.KeySwitchKey] = [:]
        for element in config.galoisElements where !galoisKeys.keys.contains(element) {
            let switchedKey = try secretKey.poly.applyGalois(element: element)
            galoisKeys[element] = try _generateKeySwitchKey(
                context: context,
                currentKey: switchedKey,
                targetKey: secretKey)
        }
        var galoisKey: GaloisKey?
        if !galoisKeys.isEmpty {
            galoisKey = GaloisKey(keys: galoisKeys)
        }
        var relinearizationKey: _RelinearizationKey<Self>?
        if config.hasRelinearizationKey {
            relinearizationKey = try Self.generateRelinearizationKey(context: context, secretKey: secretKey)
        }
        return EvaluationKey(galoisKey: galoisKey, relinearizationKey: relinearizationKey)
    }

    @inlinable
    static func generateRelinearizationKey(context: Context,
                                           secretKey: borrowing SecretKey<Self>) throws
        -> _RelinearizationKey<Self>
    {
        let s2 = secretKey.poly * secretKey.poly
        let keySwitchingKey = try _generateKeySwitchKey(context: context, currentKey: s2, targetKey: secretKey)
        return _RelinearizationKey(keySwitchKey: keySwitchingKey)
    }

    ///  Generate the key switching key from current key to target key.
    @inlinable
    public static func _generateKeySwitchKey(context: Context,
                                             currentKey: consuming PolyRq<T, Eval>,
                                             targetKey: borrowing SecretKey<Bfv<T>>) throws -> _KeySwitchKey<Bfv<T>>
    {
        guard let keyModulus = context.coefficientModuli.last else {
            throw HeError.invalidEncryptionParameters(context.encryptionParameters)
        }
        let ciphertextContext = context.ciphertextContext
        let degree = context.degree
        var ciphers: [Ciphertext<Bfv<T>, Eval>] = []
        ciphers.reserveCapacity(ciphertextContext.moduli.count)
        for (rowIndex, modulus) in ciphertextContext.reduceModuli.enumerated() {
            let keySwitchKeyCoeff = try Bfv<T>.encryptZero(
                for: context,
                using: targetKey,
                with: context.keySwitchingContexts[targetKey.moduli.count - 2])
            var keySwitchKey = try keySwitchKeyCoeff.forwardNtt()

            let modulusProduct = MultiplyConstantModulus(
                multiplicand: modulus.reduce(keyModulus),
                modulus: modulus.modulus,
                variableTime: true)
            for columnIndex in 0..<degree {
                let prod = modulusProduct.multiplyMod(currentKey.data[rowIndex, columnIndex])
                keySwitchKey.polys[0].data[rowIndex, columnIndex] = keySwitchKey.polys[0].data[rowIndex, columnIndex]
                    .addMod(prod, modulus: modulus.modulus)
            }
            ciphers.append(keySwitchKey)
        }
        // zeroize currentKey and drop it
        currentKey.zeroize()
        _ = consume currentKey

        return KeySwitchKey(context: context, ciphertexts: ciphers)
    }

    /// Precomputes the digit decomposition and NTT of a polynomial for hoisted key switching.
    ///
    /// When applying multiple Galois automorphisms to the same ciphertext (e.g., in baby-step
    /// rotations), the digit decomposition and forward NTT of `polys[1]` are identical across
    /// all automorphisms. This method precomputes them once for reuse with
    /// ``_computeKeySwitchingUpdateFromDecomposition``.
    /// - Parameters:
    ///   - context: Context for HE computation.
    ///   - target: The polynomial to decompose (typically `ciphertext.polys[1]`).
    /// - Returns: Precomputed decomposition for use with hoisted key switching.
    /// - Throws: Error upon failure to decompose.
    @inlinable
    public static func decomposeForKeySwitching(
        context: Context,
        target: PolyRq<Scalar, CanonicalCiphertextFormat>) throws -> KeySwitchDecomposition<Scalar>
    {
        let degree = target.degree
        let decomposeModuliCount = target.moduli.count
        let rnsModuliCount = decomposeModuliCount &+ 1

        let keySwitchingContext = context.keySwitchingContexts[target.moduli.count - 1]
        guard let topKeySwitchingContext = context.keySwitchingContexts.last else {
            throw HeError.invalidContext(context)
        }
        let keySwitchingModuli = keySwitchingContext.reduceModuli

        let targetCoeff = try target.convertToCoeffFormat()

        // Store UNREDUCED coefficient-domain digits per decomposition index.
        // The Galois automorphism must be applied BEFORE reduction (they don't commute
        // for negated coefficients), so we store one copy per decomposeIndex.
        var coeffDigits = [Scalar](repeating: 0, count: decomposeModuliCount &* degree)

        for decomposeIndex in 0..<decomposeModuliCount {
            let bufferSlice = targetCoeff.poly(rnsIndex: decomposeIndex)
            let offset = decomposeIndex &* degree
            for i in 0..<degree {
                coeffDigits[offset &+ i] = bufferSlice[i]
            }
        }

        return KeySwitchDecomposition(
            nttDigits: coeffDigits, // unreduced coeff digits, indexed by decomposeIndex
            degree: degree,
            decomposeModuliCount: decomposeModuliCount,
            rnsModuliCount: rnsModuliCount)
    }

    /// Computes key-switching update using a precomputed decomposition and a Galois element.
    ///
    /// This is the hoisted variant of ``_computeKeySwitchingUpdate`` — it reuses precomputed
    /// NTT'd digits and applies the Galois permutation in NTT domain, saving the per-automorphism
    /// decomposition and forward NTT cost.
    /// - Parameters:
    ///   - context: Context for HE computation.
    ///   - decomposition: Precomputed decomposition from ``_decomposeForKeySwitching``.
    ///   - element: Galois element for the automorphism to apply.
    ///   - keySwitchingKey: Key-switching key for this Galois element.
    /// - Returns: The key-switching update polynomials.
    /// - Throws: Error upon failure to compute the update.
    @inlinable
    public static func _computeKeySwitchingUpdateFromDecomposition(
        context: Context,
        decomposition: KeySwitchDecomposition<Scalar>,
        element: Int,
        keySwitchingKey: Self.KeySwitchKey) throws -> [PolyRq<Scalar, CanonicalCiphertextFormat>]
    {
        let degree = decomposition.degree
        let decomposeModuliCount = decomposition.decomposeModuliCount
        let rnsModuliCount = decomposition.rnsModuliCount

        let keySwitchingContext = context.keySwitchingContexts[decomposeModuliCount - 1]
        guard let topKeySwitchingContext = context.keySwitchingContexts.last else {
            throw HeError.invalidContext(context)
        }
        let keySwitchingModuli = keySwitchingContext.reduceModuli

        let keyComponentCount = keySwitchingKey.ciphertexts[0].polys.count
        let polys = [PolyRq<Scalar, Eval>](
            repeating: PolyRq.zero(context: keySwitchingContext),
            count: keyComponentCount)
        var ciphertextProd: EvalCiphertext = try Ciphertext(context: context,
                                                            polys: polys,
                                                            correctionFactor: 1)

        // Build the INVERSE Coeff-domain Galois permutation.
        // The iterator maps input index i → output index perm(i) with sign(i).
        // We need: output[j] = sign(inv(j)) * input[inv(j)].
        var inversePermutation = [Int](repeating: 0, count: degree)
        var inverseNegate = [Bool](repeating: false, count: degree)
        var galoisIter = GaloisCoeffIterator(degree: degree, galoisElement: element)
        for inputIndex in 0..<degree {
            guard let (negate, outputIndex) = galoisIter.next() else {
                preconditionFailure("GaloisCoeffIterator out of range")
            }
            inversePermutation[outputIndex] = inputIndex
            inverseNegate[outputIndex] = negate
        }

        let keyCiphers = keySwitchingKey.ciphertexts
        var permutedDigit = [Scalar](repeating: 0, count: degree)

        for rnsIndex in 0..<rnsModuliCount {
            let keyIndex = rnsIndex == rnsModuliCount &- 1
                ? topKeySwitchingContext.moduli.count &- 1 : rnsIndex
            let keyModulus = keySwitchingModuli[rnsIndex]

            var accumulator = Array2d(
                data: [T.DoubleWidth](
                    repeating: 0,
                    count: keyComponentCount &* degree),
                rowCount: keyComponentCount,
                columnCount: degree)

            for decomposeIndex in 0..<decomposeModuliCount {
                let qKeyJ = keySwitchingModuli[decomposeIndex]
                // Apply Galois permutation to unreduced digit (using original modulus for negation)
                let digitOffset = decomposeIndex &* degree
                for j in 0..<degree {
                    let value = decomposition.nttDigits[digitOffset &+ inversePermutation[j]]
                    if inverseNegate[j] {
                        permutedDigit[j] = value.negateMod(modulus: qKeyJ.modulus)
                    } else {
                        permutedDigit[j] = value
                    }
                }
                // Reduce mod keyModulus if needed (AFTER Galois, matching original order)
                if qKeyJ.modulus > keyModulus.modulus {
                    for index in permutedDigit.indices {
                        permutedDigit[index] = keyModulus.reduce(permutedDigit[index])
                    }
                }
                // Forward NTT the permuted digit
                try permutedDigit.withUnsafeMutableBufferPointer { bufferPtr in
                    try topKeySwitchingContext.forwardNtt(
                        // swiftlint:disable:next force_unwrapping
                        dataPtr: bufferPtr.baseAddress!,
                        modulus: keyModulus.modulus)
                }

                for (index, poly) in keyCiphers[decomposeIndex].polys.enumerated() {
                    let accIndex = poly.data.index(row: index, column: 0)
                    let polyIndex = poly.data.index(row: keyIndex, column: 0)
                    let polySpan = poly.data.data.span
                    for columnIndex in 0..<degree {
                        let prod = permutedDigit[columnIndex]
                            .multipliedFullWidth(by: polySpan[polyIndex &+ columnIndex])
                        accumulator[accIndex &+ columnIndex] &+= T.DoubleWidth(prod)
                    }
                }
            }
            let prodIndex = ciphertextProd.polys[0].data.index(row: rnsIndex, column: 0)
            for rowIndex in ciphertextProd.polys.indices {
                let accIndex = accumulator.index(row: rowIndex, column: 0)
                var ciphertextProdSpan = ciphertextProd.polys[rowIndex].data.data.mutableSpan
                for columnIndex in 0..<degree {
                    ciphertextProdSpan[prodIndex &+ columnIndex] = keyModulus
                        .reduce(accumulator[accIndex &+ columnIndex])
                }
            }
        }
        var canonicalProd = try ciphertextProd.convertToCanonicalFormat()
        try canonicalProd.modSwitchDown()
        return canonicalProd.polys
    }

    /// Applies a Galois automorphism using a precomputed key-switching decomposition (hoisted).
    ///
    /// This is faster than ``applyGalois(ciphertext:element:using:)`` when applying multiple
    /// automorphisms to the same ciphertext, because the digit decomposition and forward NTT
    /// of `polys[1]` are computed only once via ``_decomposeForKeySwitching``.
    /// - Parameters:
    ///   - ciphertext: Ciphertext to transform. Must be the same ciphertext used for the decomposition.
    ///   - element: Galois element of the transformation.
    ///   - decomposition: Precomputed decomposition from ``_decomposeForKeySwitching``.
    ///   - evaluationKey: Evaluation key containing the Galois element.
    /// - Throws: Error upon failure to apply the transformation.
    @inlinable
    public static func applyGaloisHoisted(
        ciphertext: inout CanonicalCiphertext,
        element: Int,
        decomposition: KeySwitchDecomposition<Scalar>,
        using evaluationKey: EvaluationKey<Bfv<T>>) throws
    {
        precondition(ciphertext.polys.count == 2, "ciphertext must have two polys when applying galois")
        precondition(
            ciphertext.correctionFactor == 1,
            "BFV Galois automorphisms not implemented for correction factor not equal to 1")
        guard let galoisKey = evaluationKey.galoisKey else {
            throw HeError.missingGaloisKey
        }
        guard let keySwitchingKey = galoisKey.keys[element] else {
            throw HeError.missingGaloisElement(element: element)
        }
        ciphertext.polys[0] = ciphertext.polys[0].applyGalois(element: element)
        let update = try Self._computeKeySwitchingUpdateFromDecomposition(
            context: ciphertext.context,
            decomposition: decomposition,
            element: element,
            keySwitchingKey: keySwitchingKey)
        ciphertext.polys[0] += update[0]
        ciphertext.polys[1] = update[1]
        ciphertext.clearSeed()
    }

    /// Computes the key-switching update of a target polynomial.
    ///
    /// We use hybrid key-switching from Appendix B.2.3 of <https://eprint.iacr.org/2021/204.pdf>, with:
    /// * `alpha = 1`, i.e., a single key-switching modulus
    /// * The HPS trick from Appendix B.2.1.
    ///
    /// To switch the key of 2-polynomial ciphertext `[c0, c1]` from secret key `sA` to another secret key `sB`, we need
    /// to set `c0 := c0 + c1 * ksk.p0`, and `c1 := c1 * ksk.p1`, where `ksk.p0` is the 0'th polynomial in the
    /// key-switching key.
    /// This function computes `c1 * ksk.p0` and `c1 * ksk.p1`.
    /// - Parameters:
    ///   - context: Context for HE computation
    ///   - target: The polynomial to perform key-switching on. The paper calls this `D_{Q_i}(a)`.
    ///   - keySwitchingKey: keySwitchingKey. The paper calls this `P_{Q_i}(a)`
    /// - Returns: The key-switching update for a 2-polynomial ciphertext.
    /// - Throws: Error upon failure to compute key-switching update.
    /// - seealso: ``Bfv/generateEvaluationKey(context:config:using:)``.
    @inlinable
    public static func _computeKeySwitchingUpdate(
        context: Context,
        target: PolyRq<Scalar, CanonicalCiphertextFormat>,
        keySwitchingKey: Self.KeySwitchKey) throws -> [PolyRq<Scalar, CanonicalCiphertextFormat>]
    {
        //  The implementation loosely follows the outline on page 36 of <https://eprint.iacr.org/2021/204.pdf>.
        // The inner product is computed in an extended base `q_0, q_1, ..., q_l, q_{ks}`, where `q_{ks}` is the special
        // key-switching modulus.

        let degree = target.degree
        let decomposeModuliCount = target.moduli.count
        let rnsModuliCount = decomposeModuliCount &+ 1

        let keySwitchingContext = context.keySwitchingContexts[target.moduli.count - 1]
        guard let topKeySwitchingContext = context.keySwitchingContexts.last else {
            throw HeError.invalidContext(context)
        }
        let keySwitchingModuli = keySwitchingContext.reduceModuli

        let keyComponentCount = keySwitchingKey.ciphertexts[0].polys.count
        let polys = [PolyRq<Scalar, Eval>](
            repeating: PolyRq.zero(context: keySwitchingContext),
            count: keyComponentCount)
        var ciphertextProd: EvalCiphertext = try Ciphertext(context: context,
                                                            polys: polys,
                                                            correctionFactor: 1)
        let targetCoeff = try target.convertToCoeffFormat()

        let keyCiphers = keySwitchingKey.ciphertexts
        for rnsIndex in 0..<rnsModuliCount {
            let keyIndex = rnsIndex == rnsModuliCount &- 1 ? topKeySwitchingContext.moduli.count &- 1 : rnsIndex
            let keyModulus = keySwitchingModuli[rnsIndex]

            // Use lazy accumulator to minimize modular reductions
            var accumulator = Array2d(
                data: [T.DoubleWidth](
                    repeating: 0,
                    count: keyComponentCount &* degree),
                rowCount: keyComponentCount,
                columnCount: degree)
            var accWriteSpan = accumulator.data.mutableSpan

            for decomposeIndex in 0..<decomposeModuliCount {
                let qKeyJ = keySwitchingModuli[decomposeIndex]
                var bufferSlice = targetCoeff.poly(rnsIndex: decomposeIndex)
                if qKeyJ.modulus > keyModulus.modulus {
                    for index in bufferSlice.indices {
                        bufferSlice[index] = keyModulus.reduce(bufferSlice[index])
                    }
                }

                try bufferSlice.withUnsafeMutableBufferPointer { bufferPtr in
                    try topKeySwitchingContext.forwardNtt(
                        // swiftlint:disable:next force_unwrapping
                        dataPtr: bufferPtr.baseAddress!,
                        modulus: keyModulus.modulus)
                }
                for (index, poly) in keyCiphers[decomposeIndex].polys.enumerated() {
                    let accIndex = poly.data.index(row: index, column: 0)
                    let polyIndex = poly.data.index(row: keyIndex, column: 0)
                    let polySpan = poly.data.data.span
                    for columnIndex in 0..<degree {
                        let prod = bufferSlice[columnIndex]
                            .multipliedFullWidth(by: polySpan[polyIndex &+ columnIndex])
                        // Overflow avoided by `maxLazyProductAccumulationCount()` check during context
                        // initialization
                        accWriteSpan[accIndex &+ columnIndex] &+= T.DoubleWidth(prod)
                    }
                }
            }
            let accReadSpan = accumulator.data.span
            let prodIndex = ciphertextProd.polys[0].data.index(row: rnsIndex, column: 0)
            for rowIndex in ciphertextProd.polys.indices {
                let accIndex = accumulator.index(row: rowIndex, column: 0)
                var ciphertextProdSpan = ciphertextProd.polys[rowIndex].data.data.mutableSpan
                for columnIndex in 0..<degree {
                    ciphertextProdSpan[prodIndex &+ columnIndex] = keyModulus
                        .reduce(accReadSpan[accIndex &+ columnIndex])
                }
            }
        }
        var canonicalProd = try ciphertextProd.convertToCanonicalFormat()
        // Drop the special modulus
        try canonicalProd.modSwitchDown()
        return canonicalProd.polys
    }
}
