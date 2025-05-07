import cupy as cp

def ang2pix_nest(nside, theta, phi):
    """GPU version of HEALPix ang2pix (NESTED scheme)"""
    z = cp.cos(theta)
    za = cp.abs(z)
    tt = phi * (2 / cp.pi)

    npix = 12 * nside**2
    pix = cp.zeros_like(theta, dtype=cp.int32)

    # Equatorial region
    eq_mask = (za <= 2.0 / 3.0)
    temp1 = nside * (0.5 + tt[eq_mask])
    temp2 = nside * (z[eq_mask] * 0.75)
    jp = (temp1 - temp2).astype(cp.int32)
    jm = (temp1 + temp2).astype(cp.int32)
    ir = nside + 1 + jp - jm
    kshift = 1 - (ir & 1)
    ip = (jp + jm - nside + kshift + 1) // 2
    ip = ip % (4 * nside)
    ip = cp.where(ip < 0, ip + 4 * nside, ip)
    face_num = (ir - 1) // nside + 4
    pix[eq_mask] = nside * nside * face_num + ip

    # North polar cap
    north_mask = (z > 2.0 / 3.0)
    tp = tt[north_mask] % 1.0
    tmp = nside * cp.sqrt(3 * (1 - z[north_mask]))
    jp = (tp * tmp).astype(cp.int32)
    jm = ((1.0 - tp) * tmp).astype(cp.int32)
    ir = jp + jm
    ip = (jp - jm + ir + 1) // 2
    face_num = ir
    pix[north_mask] = face_num * nside * nside + ip

    # South polar cap
    south_mask = (z < -2.0 / 3.0)
    tp = tt[south_mask] % 1.0
    tmp = nside * cp.sqrt(3 * (1 + z[south_mask]))
    jp = (tp * tmp).astype(cp.int32)
    jm = ((1.0 - tp) * tmp).astype(cp.int32)
    ir = jp + jm
    ip = (jp - jm + ir + 1) // 2
    face_num = 11 - ir
    pix[south_mask] = face_num * nside * nside + ip

    return cp.clip(pix, 0, npix - 1)
