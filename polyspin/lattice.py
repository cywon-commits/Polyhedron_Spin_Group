"""Periodic square (d=2) and simple-cubic (d=3) lattices.

Neighbour table convention: columns 0..d-1 are the +x,+y,(+z) neighbours and
columns d..2d-1 the matching -x,-y,(-z) neighbours.  Every bond is therefore
listed once among the first d columns, which the energy routines rely on.
"""
import numpy as np


class Lattice:
    def __init__(self, dim, size):
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        if size < 4 or size % 2:
            raise ValueError("size must be even and >= 4 (bipartite, no double bonds)")
        self.dim, self.size = dim, size
        self.N = size ** dim
        self.z = 2 * dim
        L = size
        coords = np.array(np.unravel_index(np.arange(self.N), (L,) * dim, order="F")).T
        self.coords = coords  # coords[:, 0] is x (fastest index)
        strides = np.array([L ** k for k in range(dim)])

        def shift(ax, d):
            c = coords.copy()
            c[:, ax] = (c[:, ax] + d) % L
            return c @ strides

        self.nbr = np.stack([shift(ax, +1) for ax in range(dim)]
                            + [shift(ax, -1) for ax in range(dim)], axis=1).astype(np.int64)
        self.sublattice = (coords.sum(axis=1) % 2).astype(np.int64)
        self.plaquettes = self._plaquettes(shift)

    def _plaquettes(self, shift):
        """(P,4) site indices a,b,c,d, counter-clockwise in each coordinate plane."""
        idx = np.arange(self.N)
        planes = [(0, 1)] if self.dim == 2 else [(0, 1), (1, 2), (2, 0)]
        out = []
        for u, v in planes:
            a = idx
            b = self.nbr[a, u]
            c = self.nbr[b, v]
            d = self.nbr[a, v]
            out.append(np.stack([a, b, c, d], axis=1))
        return np.concatenate(out).astype(np.int64)

    def site_index(self, *xyz):
        return int(sum(c * self.size ** k for k, c in enumerate(xyz)))
