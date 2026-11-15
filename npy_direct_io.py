"""Sequential-file access to C-contiguous ``.npy`` arrays.

The files remain ordinary NumPy arrays, but samples are read and written with
``seek`` plus regular file I/O rather than through an OS memory mapping.  This
keeps the prepared-dataset format usable on filesystems without ``mmap``.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np


def _read_header(stream: BinaryIO) -> tuple[tuple[int, ...], np.dtype, int]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"unsupported .npy format version {version}")
    if fortran_order:
        raise ValueError("only C-contiguous .npy arrays are supported")
    return tuple(int(size) for size in shape), np.dtype(dtype), stream.tell()


class NpyDirectWriter:
    """Write complete samples of a fixed-shape C-contiguous ``.npy`` array."""

    def __init__(
        self, path: Path, *, dtype: np.dtype | None = None,
        shape: tuple[int, ...] | None = None, create: bool,
    ) -> None:
        self.path = Path(path)
        self._stream: BinaryIO | None = None
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.shape = tuple(int(size) for size in shape)
            self.dtype = np.dtype(dtype)
            with self.path.open("w+b") as stream:
                header = {
                    "descr": np.lib.format.dtype_to_descr(self.dtype),
                    "fortran_order": False, "shape": self.shape,
                }
                np.lib.format.write_array_header_2_0(stream, header)
                self.data_offset = stream.tell()
                stream.truncate(self.data_offset + self.nbytes)
        else:
            with self.path.open("rb") as stream:
                self.shape, self.dtype, self.data_offset = _read_header(stream)

        self.sample_shape = self.shape[1:]
        self.sample_nbytes = int(np.prod(self.sample_shape, dtype=np.int64)) * self.dtype.itemsize

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize

    def _open(self) -> BinaryIO:
        if self._stream is None:
            self._stream = self.path.open("r+b", buffering=0)
        return self._stream

    def write(self, start: int, values: np.ndarray) -> None:
        values = np.ascontiguousarray(np.asarray(values, dtype=self.dtype))
        stream = self._open()
        stream.seek(self.data_offset + start * self.sample_nbytes)
        remaining = memoryview(values).cast("B")
        while remaining:
            written = stream.write(remaining)
            if written is None or written <= 0:
                raise OSError(f"failed to write {self.path}")
            remaining = remaining[written:]

    def __setitem__(self, index: int | slice, values: np.ndarray) -> None:
        if isinstance(index, slice):
            start, _, step = index.indices(self.shape[0])
            if step != 1:
                raise ValueError("only contiguous sample slices are supported")
            self.write(start, values)
            return
        normalized = index + self.shape[0] if index < 0 else index
        self.write(normalized, np.asarray(values)[None, ...])

    def flush(self) -> None:
        if self._stream is not None:
            self._stream.flush()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __del__(self) -> None:
        self.close()

