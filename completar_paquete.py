"""Añade el checkpoint original verificado y crea el ZIP definitivo de entrega."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--salida',type=Path,default=ROOT.parent/'TFM_Reproduccion_UAX_Completo.zip')
    args=parser.parse_args()
    cfg=json.loads((ROOT/'configs/final_evaluation_primary.json').read_text())
    expected=cfg['system']['checkpoint_sha256'];source=args.checkpoint.expanduser().resolve()
    if not source.is_file():raise FileNotFoundError(source)
    if digest(source)!=expected:raise RuntimeError('El archivo no coincide con el checkpoint principal congelado; no se genera el ZIP.')
    output=args.salida.expanduser().resolve()
    if output.exists():raise FileExistsError('El ZIP de destino ya existe. Elija otro nombre para conservarlo.')
    if output.is_relative_to(ROOT):raise RuntimeError('Guarde el ZIP fuera de la carpeta del paquete.')
    target=ROOT/cfg['system']['checkpoint'];target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and digest(target)!=expected:raise RuntimeError('Ya existe otro archivo diferente en la ruta del checkpoint.')
    if source!=target.resolve() and not target.exists():shutil.copy2(source,target)
    assert digest(target)==expected
    estado={'checkpoint_principal_incluido':True,'checkpoint_sha256':expected,'imagenes_originales_incluidas':False,'preparacion_imagenes':'Descargar los archivos oficiales y ejecutar preparar_imagenes.py sin regenerar particiones.','verificacion_inferencia_modelo':'No ejecutada por este empaquetador. Requiere dependencias, GPU CUDA e imágenes.','pesos_otras_arquitecturas_incluidos':False}
    (ROOT/'ESTADO_PAQUETE.json').write_text(json.dumps(estado,ensure_ascii=False,indent=2)+'\n')
    paths=sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.zip'] and not p.is_relative_to(ROOT/'data/raw'))
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths if p.name!='ARCHIVOS_ENTREGA.json'}
    (ROOT/'ARCHIVOS_ENTREGA.json').write_text(json.dumps({'archivos':hashes},ensure_ascii=False,indent=2)+'\n')
    from verificar_entrega import verificar
    report=verificar()
    if not report['correcto']:raise RuntimeError(json.dumps(report,ensure_ascii=False))
    if ROOT/'ARCHIVOS_ENTREGA.json' not in paths:paths.append(ROOT/'ARCHIVOS_ENTREGA.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in paths:z.write(p,'TFM_Reproduccion_UAX/'+str(p.relative_to(ROOT)))
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None:raise RuntimeError('Error de integridad ZIP')
    print('ZIP creado:',output)
    print('SHA-256:',digest(output))
    print('El paquete incluye el checkpoint principal original. Imágenes y pesos de las otras arquitecturas se obtienen aparte según INSTRUCCIONES.txt.')
if __name__=='__main__':main()
