import numpy as np

C1_32 = 0xCC9E2D51
C2_32 = 0x1B873593


def rotl32(x: np.uint32, r: np.uint32) -> np.uint32:
    return (x << r) | (x >> (32 - r))


def fmix32(h: np.uint32) -> np.uint32:
    h ^= h >> 16
    h *= 0x85EBCA6B
    h ^= h >> 13
    h *= 0xC2B2AE35
    h ^= h >> 16
    return h


def murmurhash3_x86_32(key: bytes, seed: np.uint32) -> np.uint32:
    data = np.frombuffer(key, dtype=np.uint8)
    nblocks = len(data) // 4

    h1 = seed

    # ---------- body ----------

    for i in range(nblocks):
        k1 = np.frombuffer(data[i * 4 : (i + 1) * 4], dtype=np.uint32)[0]

        k1 *= C1_32
        k1 = rotl32(k1, 15)
        k1 *= C2_32

        h1 ^= k1
        h1 = rotl32(h1, 13)
        h1 = h1 * 5 + 0xE6546B64

    # ---------- tail ----------

    tail = data[nblocks * 4 :]
    k1 = 0

    for i, byte in enumerate(tail):
        k1 ^= np.left_shift(byte, i * 8)

    k1 *= C1_32
    k1 = rotl32(k1, 15)
    k1 *= C2_32
    h1 ^= k1

    # ---------- finalization ----------

    h1 ^= len(data)
    h1 = fmix32(h1)

    return h1
