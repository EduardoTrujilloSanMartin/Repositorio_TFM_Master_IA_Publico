"""Recalcula las métricas guardadas sin imágenes, pesos, inferencia ni entrenamiento."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score

ROOT=Path(__file__).resolve().parent
CLASES=['akiec','bcc','bkl','df','mel','nv','vasc']
FINAL=ROOT/'experiments/final_evaluation/primary_seed43_epoch21_20261021_155351'
def ece(p,y):
    confianza=p.max(axis=1);aciertos=p.argmax(axis=1)==y
    indices=np.minimum((confianza*15).astype(int),14)
    return sum(float((indices==i).mean()*abs(aciertos[indices==i].mean()-confianza[indices==i].mean())) for i in range(15) if (indices==i).any())
def main():
    df=pd.read_csv(FINAL/'test_predictions_complete.csv',float_precision='round_trip')
    cfg=json.loads((ROOT/'configs/final_evaluation_primary.json').read_text())
    y=df.true_index.to_numpy(int)
    logits=df[[f'logit_{c}' for c in CLASES]].to_numpy(float)
    temperatura=cfg['frozen_calibration']['temperature']
    escalados=logits/temperatura
    p=np.exp(escalados-logsumexp(escalados,axis=1,keepdims=True))
    pred=p.argmax(axis=1);aceptadas=p.max(axis=1)>=cfg['frozen_calibration']['threshold'];errores=pred!=y
    original=df[[f'probability_calibrated_{c}' for c in CLASES]].to_numpy(float)
    assert np.max(np.abs(p-original))<1e-8
    assert np.array_equal(pred,df.predicted_index.to_numpy(int))
    assert np.array_equal(aceptadas,df.accepted.to_numpy(bool))
    observado={'n':len(y),'f1_macro':float(f1_score(y,pred,average='macro')),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
       'accuracy':float(accuracy_score(y,pred)),'auroc_macro_ovr':float(roc_auc_score(y,p,multi_class='ovr',average='macro'))}
    esperado=json.loads((FINAL/'performance_metrics.json').read_text())['image_level']
    for clave in esperado:assert np.isclose(observado[clave],esperado[clave],atol=1e-10,rtol=0),clave
    resumen={'rendimiento':observado,'sensibilidad_melanoma':float((pred[y==4]==4).mean()),'log_loss_calibrada':float(np.mean(logsumexp(escalados,axis=1)-escalados[np.arange(len(y)),y])),
       'brier_calibrado':float(np.mean(np.sum((p-np.eye(7)[y])**2,axis=1))),'ece15_calibrado':ece(p,y),
       'aceptadas':int(aceptadas.sum()),'rechazadas':int((~aceptadas).sum()),'cobertura':float(aceptadas.mean()),'error_aceptadas':float(errores[aceptadas].mean()),'error_rechazadas':float(errores[~aceptadas].mean())}
    assert resumen['aceptadas']==1343 and resumen['rechazadas']==156
    print(json.dumps(resumen,ensure_ascii=False,indent=2))
    print('Coincidencia con las métricas publicadas. No se ha realizado inferencia ni actualizado el modelo.')
if __name__=='__main__':main()
