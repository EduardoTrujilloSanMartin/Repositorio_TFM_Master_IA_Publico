"""Comprueba el paquete y los artefactos congelados sin cargar el modelo."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def verificar(imagenes=False):
    fallos=[]
    inventario=json.loads((ROOT/'ARCHIVOS_ENTREGA.json').read_text())
    for nombre,huella in inventario['archivos'].items():
        p=ROOT/nombre
        if not p.is_file():fallos.append('Falta archivo del paquete: '+nombre)
        elif sha256(p)!=huella:fallos.append('Contenido distinto del paquete: '+nombre)
    cfg=json.loads((ROOT/'configs/final_evaluation_primary.json').read_text())
    referencias=[(cfg['system']['checkpoint'],cfg['system']['checkpoint_sha256']),
        (cfg['data']['test_manifest'],cfg['data']['test_manifest_sha256']),
        (cfg['frozen_calibration']['temperature_source'],cfg['frozen_calibration']['temperature_source_sha256']),
        (cfg['frozen_calibration']['threshold_source'],cfg['frozen_calibration']['threshold_source_sha256'])]
    cal=json.loads((ROOT/'configs/calibration_abstention_primary.json').read_text())
    referencias.extend([(cal['data']['selection_manifest'],cal['data']['selection_manifest_sha256']),
        (cal['data']['calibration_manifest'],cal['data']['calibration_manifest_sha256'])])
    for nombre,huella in referencias:
        p=ROOT/nombre
        if not p.is_file():fallos.append('Falta artefacto congelado: '+nombre)
        elif sha256(p)!=huella:fallos.append('Huella congelada incompatible: '+nombre)
    total=0
    if imagenes:
        with (ROOT/'data/manifests/ham10000_v1_seed2026/all_splits.csv').open(newline='') as f:
            for fila in csv.DictReader(f):
                p=ROOT/'data'/fila['relative_path'];total+=1
                if not p.is_file():fallos.append('Falta imagen: '+fila['image_id'])
                elif sha256(p)!=fila['file_sha256']:fallos.append('Imagen distinta: '+fila['image_id'])
    return {'correcto':not fallos,'archivos_inventariados':len(inventario['archivos']),'imagenes_comprobadas':total,'fallos':fallos}
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--imagenes',action='store_true')
    resultado=verificar(parser.parse_args().imagenes)
    print(json.dumps(resultado,ensure_ascii=False,indent=2))
    raise SystemExit(0 if resultado['correcto'] else 2)
