from lightecc import LightECC
import random
import numpy as np

def message_to_point(m, curve, k=100):
    a = curve.a
    b = curve.b
    q = curve.modulo

    if (m+1)*k >= q:
        raise ValueError("message too large for F_q field of curve")

    for i in range(k):
        x = m * k + i
        y_squared = (pow(x, 3, q)+a*x+b)%q
        y = pow(y_squared, (q+1)//4, q)
        if pow(y, 2, q)==y_squared:
            return (x, y)
    
    raise ValueError(f"no point found for message in k = {k} attempts")

def point_to_message(point, k=100):
    x, _ = point
    return x//k

def KeyGen(curve):
    n = curve.n
    p = curve.G
    x = random.randint(1, n-1)
    xp = curve.double_and_add(p, x)
    pk = (xp, p, curve)
    sk = (x, curve)
    return (sk, pk)

def Encode(pk, m, r=[0]):
    m_str = ''.join(list(map(lambda x: chr(x+48), m)))
    r_str = ''.join(list(map(lambda x: chr(x+48), r)))
    xor = int(m_str, 2) ^ int(r_str, 2)

    y, p, curve = pk
    n = curve.n
    l = len(m)
    start = 0
    padded_blocks = []

    while start < l:
        sub = m[start:start+248]
        if len(sub) == 248:
            padded_blocks.append(sub)
        else:
            pad = [0]*(247-len(sub))+[1]
            padded_blocks.append(pad+sub)
        start += 248
    
    if l % 248 == 0:
        pad_block = [0] * 247 + [1]
        padded_blocks.append(pad_block)

    pows = [2**i for i in range(248)][::-1]
    res = []

    for i in range(len(padded_blocks)):
        m = sum([padded_blocks[i][j]*pows[j] for j in range(248)])
        k = random.randint(1, n-1)
        c = curve.double_and_add(p, k)
        cy = curve.double_and_add(y, k)
        pm = message_to_point(m, curve=curve)
        s = curve.add_points(cy, pm)
        res.append((c, s))
    return (res, xor)
    
def Decode(sk, x):
    res = []
    ct, xor = x
    x_sk, curve = sk
    for i in range(len(ct)):
        c, d = ct[i]
        cx = curve.double_and_add(c, x_sk)
        inv_cx = curve.negative_point(cx)
        pm = curve.add_points(d, inv_cx)
        m = point_to_message(pm)
        if (i == len(ct)-1):
            brep = list(map(int, list("{0:b}".format(m))))
            res.extend(brep[1:])
        else:
            brep = list(map(int, list("{0:0248b}".format(m))))
            res.extend(brep)
    return res

def Recover(pk, x, r):
    ct, xor = x
    r_str = ''.join(list(map(lambda x: chr(x+48), r)))
    m_int = xor ^ int(r_str, 2)
    return list(map(int, list("{0:b}".format(m_int))))

def correctness_test():
    fail = False
    for _ in range(100):
        bits = np.random.randint(low=0, high=2, size=random.randint(1, 1000))
        m = bits.tolist()
        sk, pk = KeyGen(curve)
        x = Encode(pk, m)
        d = Decode(sk, x)
        if (m != d):
            fail = True
            print("correctness error:")
            print(m)
            print(d)
    if (fail):
        print("correctness test failed: see above prints to verify")
    else:
        print("correctness test passed")

ecc = LightECC(curve_name="secp256k1")
curve = ecc.curve
bits = "100011110001101000100001110101101100111010011110111101110011110011111110010000101101110000001011100000010111110011001100111101110100011100110110010100111011010010111001110010110010110101101100011010111110101110011101101011110111101001110"
rbits = "111101111011100111100111111100100001011011100000010"
m = list(map(int, list(bits)))
r = list(map(int, list(rbits)))
sk, pk = KeyGen(curve)
x = Encode(pk, m, r)
d = Decode(sk, x)
re = Recover(pk, x, r)
print(''.join(list(map(lambda x: chr(x+48), re))) == bits)
print(''.join(list(map(lambda x: chr(x+48), d))) == bits)