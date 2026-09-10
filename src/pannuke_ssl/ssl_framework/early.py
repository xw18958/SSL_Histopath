from __future__ import annotations
import math
from typing import Any

class EarlyStopper:
    def __init__(self, *, enabled: bool, min_epochs: int, patience: int, delta: float):
        self.enabled,self.min_epochs,self.patience,self.delta=bool(enabled),int(min_epochs),int(patience),float(delta)
        self.best,self.best_epoch,self.stale=-math.inf,0,0
    def update(self, epoch:int, score:float) -> dict[str,Any]:
        if not math.isfinite(score): raise FloatingPointError("Non-finite validation score")
        improved=self.best_epoch==0 or score>=self.best+self.delta
        if improved: self.best,self.best_epoch,self.stale=float(score),int(epoch),0
        elif self.enabled and epoch>self.min_epochs: self.stale+=1
        else: self.stale=0
        return {"improved":improved,"should_stop":self.enabled and epoch>self.min_epochs and self.stale>=self.patience,"best_score":self.best,"best_epoch":self.best_epoch,"stale_monitors":self.stale}
