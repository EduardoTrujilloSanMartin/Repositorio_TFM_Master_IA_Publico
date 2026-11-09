"""Extrae y verifica imágenes oficiales conservando los manifiestos incluidos."""
import csv
import hashlib
import zipfile
from pathlib import Path,PurePosixPath

ROOT=Path(__file__).resolve().parent
ARCHIVES=ROOT/'data/raw/ham10000_v1/archives'
EXPECTED={'HAM10000_images_part_1.zip':'4639bfa73ab251610530a97c898e6e46','HAM10000_images_part_2.zip':'da43d6cc50f6613013be07e8986b384b'}
def digest(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    with (ROOT/'data/manifests/ham10000_v1_seed2026/all_splits.csv').open(newline='') as f:
        filas={row['image_id']:row for row in csv.DictReader(f)}
    assert len(filas)==10015
    vistos=set()
    for archivo,md5 in EXPECTED.items():
        source=ARCHIVES/archivo
        if not source.is_file():raise FileNotFoundError('Primero descargue los originales: python scripts/download_ham10000.py')
        if digest(source,'md5')!=md5:raise RuntimeError('ZIP oficial incompatible: '+archivo)
        with zipfile.ZipFile(source) as z:
            for info in z.infolist():
                if info.is_dir():continue
                name=PurePosixPath(info.filename).name;image_id=Path(name).stem
                if image_id not in filas or Path(name).suffix.lower()!='.jpg':continue
                row=filas[image_id];dest=ROOT/'data'/row['relative_path']
                if not dest.resolve().is_relative_to((ROOT/'data').resolve()):raise RuntimeError('Ruta fuera de data')
                if info.file_size>64*1024*1024:raise RuntimeError('Imagen mayor del límite de extracción')
                if not dest.is_file():
                    data=z.read(info)
                    if hashlib.sha256(data).hexdigest()!=row['file_sha256']:raise RuntimeError('Contenido incompatible: '+image_id)
                    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
                if digest(dest)!=row['file_sha256']:raise RuntimeError('Imagen local incompatible: '+image_id)
                vistos.add(image_id)
    if vistos!=set(filas):raise RuntimeError('No se encontraron todas las imágenes del manifiesto')
    print('10.015 imágenes extraídas/verificadas. Las particiones no se han regenerado ni modificado.')
if __name__=='__main__':main()
