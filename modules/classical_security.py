"""
Conventional secure communication baseline: ECDH key exchange + AES-GCM
encryption, used as the classical comparison point against QKD (qkd.py).
"""

import time
import json

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import config


# ============================================================
# CLASSICAL KEY ESTABLISHMENT (ECDH)
# ============================================================

def establish_classical_key():
    """
    Simulate a classical ECDH handshake between two parties (the operator
    and the recipient of the collision-warning message). Returns the
    derived AES-256 key plus overhead metrics for comparison with QKD.
    """
    start = time.time()

    # Each party generates an ephemeral key pair
    party_a_private = ec.generate_private_key(ec.SECP384R1())
    party_b_private = ec.generate_private_key(ec.SECP384R1())

    party_a_public_bytes = party_a_private.public_key().public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )
    party_b_public_bytes = party_b_private.public_key().public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )

    # Each party derives the same shared secret from the other's public key
    party_b_public = ec.EllipticCurvePublicKey.from_encoded_point(
        ec.SECP384R1(), party_b_public_bytes
    )
    party_a_public = ec.EllipticCurvePublicKey.from_encoded_point(
        ec.SECP384R1(), party_a_public_bytes
    )

    shared_secret_a = party_a_private.exchange(ec.ECDH(), party_b_public)
    shared_secret_b = party_b_private.exchange(ec.ECDH(), party_a_public)
    assert shared_secret_a == shared_secret_b, "ECDH shared secrets did not match"

    aes_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"orbital-debris-collision-warning",
    ).derive(shared_secret_a)

    runtime = time.time() - start

    return {
        "key": aes_key,
        "runtime_sec": runtime,
        "public_key_bytes_exchanged": len(party_a_public_bytes) + len(party_b_public_bytes),
    }


# ============================================================
# AES-GCM ENCRYPTION (shared by classical_security.py and qkd.py)
# ============================================================

def encrypt_message(key, plaintext):
    aesgcm = AESGCM(key)
    nonce = __import__("os").urandom(12)
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    return nonce, ciphertext


def decrypt_message(key, nonce, ciphertext):
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return plaintext.decode("utf-8")


# ============================================================
# PIPELINE
# ============================================================

def run_and_save():
    config.ensure_dirs()

    warnings_path = config.RESULTS_DIR / "warnings.txt"
    if warnings_path.exists():
        message = warnings_path.read_text().strip()
        if not message or message.startswith("No MEDIUM/HIGH"):
            message = ("=== COLLISION WARNING (sample) ===\n"
                       "No real MEDIUM/HIGH risk event on file — using a "
                       "placeholder message to demonstrate the security layer.")
    else:
        message = "=== COLLISION WARNING (sample) ===\nPlaceholder message."

    handshake = establish_classical_key()
    key = handshake["key"]

    nonce, ciphertext = encrypt_message(key, message)
    decrypted = decrypt_message(key, nonce, ciphertext)
    assert decrypted == message, "Round-trip decryption mismatch"

    result = {
        "method": "Classical (ECDH + AES-256-GCM)",
        "handshake_runtime_sec": handshake["runtime_sec"],
        "public_key_bytes_exchanged": handshake["public_key_bytes_exchanged"],
        "plaintext_bytes": len(message.encode("utf-8")),
        "ciphertext_bytes": len(ciphertext),
        "round_trip_verified": decrypted == message,
    }

    output_path = config.RESULTS_DIR / "classical_security_results.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print("=== Classical Security (ECDH + AES-GCM) ===")
    for k, v in result.items():
        print(f"{k}: {v}")
    print(f"\nResults written: {output_path}")

    return result


if __name__ == "__main__":
    run_and_save()