// Minimal utilities to decode base64-encoded .npz (zip) containing a single .npy array
// Returns a typed array and shape, and a helper to convert to nested [D][H][W]

import JSZip from 'jszip';

export async function decodeBase64ToArrayBuffer(b64: string): Promise<ArrayBuffer> {
  const binary = atob(b64);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

type ParsedNpy = { data: ArrayBuffer; dtype: string; shape: number[] };

// Very small NPY v1.0/2.0 parser for C-order arrays
function parseNPY(buf: ArrayBuffer): ParsedNpy {
  const u8 = new Uint8Array(buf);
  // magic header: \x93NUMPY
  if (!(u8[0] === 0x93 && String.fromCharCode(...u8.slice(1, 6)) === 'NUMPY'))
    throw new Error('Invalid NPY file');
  const major = u8[6];
  const minor = u8[7];
  let headerLen = 0;
  if (major === 1) {
    headerLen = u8[8] | (u8[9] << 8);
    var offset = 10;
  } else {
    headerLen = u8[8] | (u8[9] << 8) | (u8[10] << 16) | (u8[11] << 24);
    var offset = 12;
  }
  const headerStr = new TextDecoder('latin1').decode(u8.slice(offset, offset + headerLen));
  const descrMatch = headerStr.match(/'descr':\s*'([^']+)'/);
  const shapeMatch = headerStr.match(/'shape':\s*\(([^\)]*)\)/);
  const fortranMatch = headerStr.match(/'fortran_order':\s*(True|False)/);
  if (!descrMatch || !shapeMatch || !fortranMatch) throw new Error('Invalid NPY header');
  const descr = descrMatch[1];
  const fortran = fortranMatch[1] === 'True';
  if (fortran) throw new Error('Fortran order arrays not supported');
  const shape = shapeMatch[1]
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map((s) => parseInt(s, 10));
  const dataStart = offset + headerLen;
  return { data: buf.slice(dataStart), dtype: descr, shape };
}

function dtypeToTypedArray(buf: ArrayBuffer, dtype: string): Int16Array | Float32Array | Float64Array | Int32Array | Uint8Array {
  // Support common dtypes for our use-case
  switch (dtype) {
    case '<i2':
    case '|i2':
      return new Int16Array(buf);
    case '<i4':
    case '|i4':
      return new Int32Array(buf);
    case '<f4':
    case '|f4':
      return new Float32Array(buf);
    case '<f8':
    case '|f8':
      return new Float64Array(buf);
    case '|u1':
    case '<u1':
      return new Uint8Array(buf);
    default:
      throw new Error(`Unsupported dtype ${dtype}`);
  }
}

export interface DecodeNpzOptions {
  quietMissing?: boolean;
}

export async function decodeNpzTo3DArray(
  b64: string,
  key: string,
  options?: DecodeNpzOptions,
): Promise<number[][][]> {
  const ab = await decodeBase64ToArrayBuffer(b64);
  const zip = await JSZip.loadAsync(ab);
  // Find .npy entry for the specific key
  const expectedNpyFile = `${key}.npy`;
  const entry = Object.keys(zip.files).find((k) => k === expectedNpyFile);
  if (!entry) {
    if (!options?.quietMissing) {
      console.error(`NPZ key '${key}' not found. Available keys:`, Object.keys(zip.files));
    }
    throw new Error(`NPZ key '${key}' not found`);
  }
  const npyBuf = await zip.files[entry].async('arraybuffer');
  const parsed = parseNPY(npyBuf);
  const arr = dtypeToTypedArray(parsed.data, parsed.dtype);
  if (parsed.shape.length !== 3) throw new Error(`Expected 3D array, got shape ${parsed.shape}`);
  const [D, H, W] = parsed.shape.map((x) => Number(x));
  const out: number[][][] = Array.from({ length: D }, () => Array.from({ length: H }, () => Array(W).fill(0)));
  let idx = 0;
  for (let z = 0; z < D; z++) {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        // @ts-ignore
        out[z][y][x] = Number(arr[idx++]);
      }
    }
  }
  return out;
}
