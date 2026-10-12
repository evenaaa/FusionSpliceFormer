"""Version and integrity checks for distributed model resources."""
from pathlib import Path
import json
from .io import sha256

MODEL_VERSION = '1.0.0'

def tensor_identity(state):
    """Content identity independent of file encoding."""
    import hashlib
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

WEIGHTS = {'model_01.pt': 'cb02fef477553564579411c0901b780022b3a1a184cfc9ed093623a5f8da1db0', 'model_02.pt': '945b50d95b72a389aa9b4b3f247413c9971bf6d3036e7e07e9c0e2bcb2fd84a0', 'model_03.pt': 'f48c2f3b0f06afcc29dac0df55f86b071dd581597b1773e03b5d3059ce8c3d3c', 'model_04.pt': 'ed39520f54417599846e7c04608e51d7a229bdfb1a83b1a804bc70441fa798de', 'model_05.pt': '7760f467f4784cc781a5c96f2c69c8d0d34ceb800226bd4f5708f8c1722a28f3', 'model_06.pt': '3532e7b30e5a02cff618d37cee09a8d51ad24f5b925b0590f9193aa66723ea33', 'model_07.pt': '2dd923aba9449a7bc49641064e9bc0224d3add7609fe279c606bbcdb501b3428', 'model_08.pt': 'dab3215cbc14058c843f1142c4df27968d836bd28c22b9500be6832ae1adc41d', 'model_09.pt': 'f632683c46ed7472d875712cfdadd4341886fe8488d163b1ea6d8b18ef329ba5', 'model_10.pt': 'e9500c4f3ce2a9a616db1992c3fdf02cdd1001eca52d36be21fed9cb3651062e'}
PREPROCESSORS = {
    'acceptor': '925fd7140715b163209995fb52ae75f55119e3973598a6a32bd734c9f534719f',
    'donor': 'c95b07865bd4dadbd789117045ac3c6ac2a2961dcb023cc1e6160fd4f7255018',
}


def released_checkpoints(root=Path('weights')):
    root = Path(root)
    manifest = root/'models.json'
    if not manifest.is_file():
        raise ValueError('The included weights/models.json is missing. Restore weights/ from the project distribution. '
                         'The official model requires ten checkpoints; no single-model fallback is used. '
                         'Use -M only for a custom model.')
    payload = json.loads(manifest.read_text(encoding='utf-8'))
    entries = payload.get('checkpoints') if isinstance(payload, dict) else None
    expected = list(WEIGHTS)
    if entries != expected:
        raise ValueError('weights/models.json does not match the official ten-model list. '
                         'Restore the included weights/ directory, or use -M for custom models.')
    paths = [root/name for name in expected]
    for path, digest in zip(paths, WEIGHTS.values()):
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'Missing or mismatched official weight: {path}. Restore weights/ from the project distribution.')
    return paths


def validate_preprocessors(root):
    for region, digest in PREPROCESSORS.items():
        path = Path(root)/region/'preprocessor.json'
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'Missing or mismatched official preprocessor: {path}. Restore weights/ from the project distribution.')
