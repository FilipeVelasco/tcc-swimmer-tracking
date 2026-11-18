import json
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provas import da_mesma_prova, fora_com

RAIZ = Path(__file__).resolve().parent.parent
SAI = RAIZ / "saidas" / "rede2"
DEV = torch.device(os.environ.get("REDE2_DEV") or ("cuda" if torch.cuda.is_available() else "cpu"))
torch.set_num_threads(6)
JAN = 64
# corridas da base Woinoski reservadas para validação (escolhidas antes de treinar, cobrindo os quatro nados):
# crawl w15, w25, w45; borboleta w00, w30; costas w10, w40; peito w12, w42
VAL_W = {"w15", "w25", "w45", "w00", "w30", "w10", "w40", "w12", "w42"}


class Rede2(nn.Module):
    def __init__(self, T=24, larg=96, nado=False):
        super().__init__()
        c = [nn.Conv2d(3, 24, 3, 2, 1), nn.BatchNorm2d(24), nn.ReLU()] if T >= 48 else [nn.Conv2d(3, 24, 3, 1, 1), nn.BatchNorm2d(24), nn.ReLU()]
        c += [nn.Conv2d(24, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 48, 3, 2, 1), nn.BatchNorm2d(48), nn.ReLU(),
              nn.Conv2d(48, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(2), nn.Flatten()]
        self.enc = nn.Sequential(*c)
        self.nado = nn.Embedding(4, 8)
        self.ent = nn.Conv1d(256 + 8, larg, 1)
        self.tmp = nn.ModuleList([nn.Sequential(nn.Conv1d(larg, larg, 5, padding=2 * d, dilation=d), nn.BatchNorm1d(larg), nn.ReLU()) for d in (1, 2, 4, 8, 1, 2)])
        self.sai = nn.Conv1d(larg, 1, 1)
        self.cab_nado = nn.Conv1d(larg, 1, 1) if nado else None      # saída extra: o nadador está nadando neste quadro?
        self.T = T

    def forward(self, x, estilo, com_nado=False):
        """com_nado=True devolve também o logit de "está nadando" por quadro (só modelos criados com nado=True)."""
        B, T = x.shape[:2]
        d = torch.zeros_like(x)
        d[:, 1:] = x[:, 1:] - x[:, :-1]
        d2 = torch.zeros_like(x)
        d2[:, 2:] = x[:, 2:] - x[:, :-2]
        f = self.enc(torch.stack([x, 4 * d, 4 * d2], 2).reshape(B * T, 3, self.T, self.T)).reshape(B, T, -1).transpose(1, 2)
        h = self.ent(torch.cat([f, self.nado(estilo)[:, :, None].expand(B, 8, T)], 1))
        for c in self.tmp:
            h = h + c(h)
        dens = F.softplus(self.sai(h)[:, 0])
        return (dens, self.cab_nado(h)[:, 0]) if com_nado else dens


class Bloco(nn.Module):

    def __init__(self, ci, co, passo):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, 3, passo, 1)
        self.norm = nn.InstanceNorm3d(co, affine=True)

    def forward(self, x, B):
        x = self.conv(x)
        BT, C, H, W = x.shape
        x = self.norm(x.reshape(B, BT // B, C, H, W).transpose(1, 2)).transpose(1, 2).reshape(BT, C, H, W)
        return F.relu(x)


class Rede3(nn.Module):
    """Como a Rede2, com a entrada padronizada por trilha e normalização por trilha no codificador de imagem."""

    def __init__(self, T=24, larg=96):
        super().__init__()
        self.blocos = nn.ModuleList([Bloco(3, 24, 2 if T >= 48 else 1), Bloco(24, 32, 2), Bloco(32, 48, 2), Bloco(48, 64, 2)])
        self.nado = nn.Embedding(4, 8)
        self.ent = nn.Conv1d(256 + 8, larg, 1)
        self.tmp = nn.ModuleList([nn.Sequential(nn.Conv1d(larg, larg, 5, padding=2 * d, dilation=d), nn.BatchNorm1d(larg), nn.ReLU()) for d in (1, 2, 4, 8, 1, 2)])
        self.sai = nn.Conv1d(larg, 1, 1)
        self.T = T

    def forward(self, x, estilo):
        B, T = x.shape[:2]
        x = (x - x.mean((1, 2, 3), keepdim=True)) / (x.std((1, 2, 3), keepdim=True) + 1e-3)
        d = torch.zeros_like(x)
        d[:, 1:] = x[:, 1:] - x[:, :-1]
        d2 = torch.zeros_like(x)
        d2[:, 2:] = x[:, 2:] - x[:, :-2]
        f = torch.stack([x, d, d2], 2).reshape(B * T, 3, self.T, self.T)
        for b in self.blocos:
            f = b(f, B)
        f = F.adaptive_avg_pool2d(f, 2).reshape(B, T, -1).transpose(1, 2)
        h = self.ent(torch.cat([f, self.nado(estilo)[:, :, None].expand(B, 8, T)], 1))
        for c in self.tmp:
            h = h + c(h)
        return F.softplus(self.sai(h)[:, 0])


CLASSES = {"Rede2": Rede2, "Rede3": Rede3}
CFG = dict(classe="Rede2", forte=False, vel=(0.45, 2.2), nos=10, lr=None, pen=0.0, nado=0.0)


def receita():
    """Opções de treino que vão junto com os pesos no arquivo .pt (carrega_rede usa 'classe' e 'nado' para remontar a rede)."""
    return dict(classe=CFG["classe"], forte=CFG["forte"], vel=list(CFG["vel"]), nos=CFG["nos"], lr=CFG["lr"], pen=CFG["pen"], nado=CFG["nado"])


def estado_cpu(rede):
    """Pesos sempre na CPU, para o arquivo .pt abrir em máquina sem GPU (na CPU não muda nada)."""
    e = rede.state_dict()
    for k in e:
        e[k] = e[k].cpu()
    return e


def carrega_rede(arq):
    ck = torch.load(arq, weights_only=False, map_location="cpu")
    rede = CLASSES[ck.get("classe", "Rede2")](ck["T"], ck["larg"], **({"nado": True} if ck.get("nado") else {}))
    rede.load_state_dict(ck["estado"])
    rede.to(DEV)
    rede.eval()
    return rede, ck


def redimensiona(X, T):
    if X.shape[1] == T:
        return X
    return np.stack([cv2.resize(f, (T, T), interpolation=cv2.INTER_AREA) for f in X])


def carrega(T, nossos="rede_dados_48.pkl"):
    with open(RAIZ / "saidas" / "woinoski_48.pkl", "rb") as f:
        W = pickle.load(f)
    for s in W:
        s["X"] = redimensiona(s["X"], T)
    a48, a24 = RAIZ / "saidas" / nossos, RAIZ / "saidas" / "rede_dados.pkl"
    # nome errado em dados= cairia calado no rede_dados.pkl antigo: só o nome padrão pode faltar
    assert a48.exists() or nossos == "rede_dados_48.pkl", f"não existe {a48} (nome errado em dados=?)"
    N = []
    if a48.exists():                                   # recortes 48x48 já com o corpo na horizontal
        with open(a48, "rb") as f:
            N = pickle.load(f)
        for s in N:
            s["X"] = redimensiona(np.ascontiguousarray(s["X"]), T)
    elif T == 24 and a24.exists():
        with open(a24, "rb") as f:
            N = pickle.load(f)
        for s in N:
            s["X"] = np.ascontiguousarray(np.rot90(s["X"], 1, axes=(1, 2)))   # rede_dados.pkl: cabeça para cima -> corpo na horizontal
    return W, N


def reamostra(X, fator):
    n = max(8, int(len(X) / fator))
    pos = np.arange(n) * fator
    i0 = np.clip(pos.astype(int), 0, len(X) - 1)
    i1 = np.clip(i0 + 1, 0, len(X) - 1)
    w = (pos - i0)[:, None, None].astype(np.float32)
    return (1 - w) * X[i0] + w * X[i1]


def rotulo_nado(n, ev, padrao=12.0):
    r = np.full(n, -1.0, np.float32)
    if ev is None:
        return r
    if len(ev) == 0:
        r[:] = 0
        return r
    ev = np.sort(np.asarray(ev, float))
    med = float(np.median(np.diff(ev))) if len(ev) >= 3 else padrao
    dist = np.abs(np.arange(n)[:, None] - ev[None, :]).min(1)
    r[dist <= med] = 1
    r[dist >= 2 * med] = 0
    return r


def aumenta(s, rng, janela=True):
    """Devolve (recortes, densidade alvo, rótulo de nado por quadro). O rótulo não consome números aleatórios: a
    sequência de janelas sorteadas é a mesma de antes de ele existir."""
    fator = float(np.exp(rng.uniform(np.log(CFG["vel"][0]), np.log(CFG["vel"][1]))))
    X = reamostra(s["X"].astype(np.float32) / 255, fator)
    ev = None if s["ev"] is None else s["ev"] / fator
    nado = rotulo_nado(len(X), ev, 12.0 / fator)
    if janela:
        if len(X) < JAN:
            nado = np.concatenate([nado, np.zeros(JAN - len(X), np.float32)])      # quadro repetido no fim: sem nado
            X = np.concatenate([X, np.repeat(X[-1:], JAN - len(X), 0)])
        a = rng.integers(0, len(X) - JAN + 1)
        X, ev, nado = X[a:a + JAN], ev - a, nado[a:a + JAN]
    if rng.random() < 0.5:
        X = X[:, :, ::-1]
    if rng.random() < 0.5:
        X = X[:, ::-1]
    m = max(1, X.shape[1] // 12)
    if CFG["forte"]:
        T = X.shape[1]
        X = np.ascontiguousarray(np.rot90(X, int(rng.integers(4)), axes=(1, 2)))            # qualquer orientação do nadador
        if rng.random() < 0.6:                                                              # zoom e deslocamento (janela mais folgada ou mais justa)
            z = float(np.exp(rng.uniform(np.log(0.8), np.log(1.3))))
            A = np.array([[z, 0, (1 - z) * T / 2 + rng.uniform(-m, m)], [0, z, (1 - z) * T / 2 + rng.uniform(-m, m)]], np.float32)
            X = np.stack([cv2.warpAffine(f, A, (T, T), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for f in X])
        if rng.random() < 0.5:                                                              # nadador pequeno ou vídeo borrado
            r = int(rng.integers(max(6, T // 3), T))
            X = np.stack([cv2.resize(cv2.resize(f, (r, r), interpolation=cv2.INTER_AREA), (T, T), interpolation=cv2.INTER_LINEAR) for f in X])
        if rng.random() < 0.15:                                                             # nadador claro sobre água escura e o contrário
            X = 1 - X
    else:
        dx, dy = rng.integers(-m, m + 1, 2)
        X = np.roll(X, (dy, dx), (1, 2))
    X = np.clip((X - 0.5) * rng.uniform(0.7, 1.4) + 0.5 + rng.uniform(-0.15, 0.15) + rng.normal(0, 0.02, X.shape), 0, 1)
    alvo = np.zeros(len(X), np.float32)
    if ev is not None and len(ev):
        t = np.arange(len(X))[:, None]
        alvo = (np.exp(-0.5 * ((t - ev[None, :]) / 1.5) ** 2) / (1.5 * np.sqrt(2 * np.pi))).sum(1).astype(np.float32)
    return np.ascontiguousarray(X, np.float32), alvo, nado


def nadando(rede, s):
    """Probabilidade de "está nadando" em cada quadro (só modelos treinados com nado=<peso>)."""
    rede.eval()
    dev = next(rede.parameters()).device               # a entrada vai para onde a rede estiver
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(s["X"].astype(np.float32) / 255))[None].to(dev)
        return torch.sigmoid(rede(x, torch.tensor([s["estilo"]], device=dev), com_nado=True)[1][0]).cpu().numpy()


def densidade(rede, s, fator=1.0, adapta=False):
    """adapta=True: as normalizações do codificador de imagem usam as estatísticas desta trilha; só para Rede2."""
    rede.eval()
    dev = next(rede.parameters()).device               # a entrada vai para onde a rede estiver
    with torch.no_grad():
        X = reamostra(s["X"].astype(np.float32) / 255, fator) if fator != 1.0 else s["X"].astype(np.float32) / 255
        x, e = torch.from_numpy(np.ascontiguousarray(X))[None].to(dev), torch.tensor([s["estilo"]], device=dev)
        if not adapta or not hasattr(rede, "enc"):
            return rede(x, e)[0].cpu().numpy()
        estado = {k: v.clone() for k, v in rede.state_dict().items()}
        rede.enc.train()
        d = rede(x, e)[0].cpu().numpy()
        rede.load_state_dict(estado)
        rede.eval()
        return d


def f1_posicao(d, ev, tol):
    p, _ = find_peaks(d, height=0.08, distance=2)
    livres, acertos = list(ev), 0
    for x in p:
        if livres:
            j = int(np.argmin([abs(x - e) for e in livres]))
            if abs(x - livres[j]) <= tol:
                acertos += 1
                livres.pop(j)
    return acertos, len(p), len(ev)


def treina(tr, passos, T, larg, semente, log=None, inicial=None, ao_medir=None, cada=0):
    """ao_medir(rede, passo): chamada a cada 'cada' passos (curva de ajuste); não consome números aleatórios, então o
    treino é idêntico com ou sem ela."""
    rng = np.random.default_rng(semente)
    torch.manual_seed(semente)
    ev = [s for s in tr if s["ev"] is not None]
    tot = [s for s in tr if s["ev"] is None]
    grupos = sorted({s["grupo"] for s in ev})
    por_grupo = {g: [s for s in ev if s["grupo"] == g] for g in grupos}
    nossos = [g for g in grupos if not g.startswith("w")]
    rede = CLASSES[CFG["classe"]](T, larg, **({"nado": True} if CFG["nado"] > 0 else {}))
    if inicial:                                        # ajuste fino a partir de um modelo já treinado
        # com a saída extra de nado, o modelo inicial pode não tê-la: ela começa do zero e o resto vem do arquivo
        falta = rede.load_state_dict(torch.load(inicial, weights_only=False, map_location="cpu")["estado"], strict=CFG["nado"] <= 0)
        assert all(k.startswith("cab_nado.") for k in falta.missing_keys) and not falta.unexpected_keys, falta
    rede.to(DEV)                                       # a rede nasce na CPU (mesmos pesos iniciais em qualquer máquina) e só então vai para a GPU
    opt = torch.optim.AdamW(rede.parameters(), 2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, CFG["lr"] or (1e-3 if inicial else 3e-3), total_steps=passos)
    t0 = time.time()
    for p in range(passos):
        rede.train()
        lote = []
        for k in range(32):
            g = nossos[rng.integers(len(nossos))] if (nossos and k < CFG["nos"]) else grupos[rng.integers(len(grupos))]
            lote.append(por_grupo[g][rng.integers(len(por_grupo[g]))])
        dados = [aumenta(s, rng) for s in lote]
        x = torch.from_numpy(np.stack([d[0] for d in dados])).to(DEV)
        y = torch.from_numpy(np.stack([d[1] for d in dados])).to(DEV)
        est = torch.tensor([s["estilo"] for s in lote], device=DEV)
        if CFG["nado"] > 0:
            pr, lg = rede(x, est, com_nado=True)
        else:
            pr = rede(x, est)
        perda = 30 * F.mse_loss(pr, y) + 0.3 * (pr.sum(1) - y.sum(1)).abs().mean()
        if CFG["pen"] > 0 or CFG["nado"] > 0:
            rn = torch.from_numpy(np.stack([d[2] for d in dados])).to(DEV)
            if CFG["pen"] > 0:                         # qualquer sinal em quadro sem nado custa, mesmo pequeno (a perda quadrática quase não cobra sinal baixo)
                perda = perda + CFG["pen"] * (pr * (rn == 0)).mean()
            if CFG["nado"] > 0:                        # saída extra: nado x sem nado, quadro a quadro; o "sem nado" é raro e pesa 3
                v = rn >= 0
                perda = perda + CFG["nado"] * F.binary_cross_entropy_with_logits(lg[v], rn[v], weight=1.0 + 2.0 * (rn[v] == 0).float())
        if tot:
            s = tot[rng.integers(len(tot))]
            X = aumenta(s, rng, janela=False)[0]
            pt = rede(torch.from_numpy(X)[None].to(DEV), torch.tensor([s["estilo"]], device=DEV)).sum()
            perda = perda + 0.6 * (pt - s["total"]).abs() / max(4.0, s["total"])
        opt.zero_grad()
        perda.backward()
        nn.utils.clip_grad_norm_(rede.parameters(), 2.0)
        opt.step()
        sched.step()
        if log and (p + 1) % 500 == 0:
            print(f"   passo {p + 1}/{passos} perda {float(perda):.3f} ({time.time() - t0:.0f} s)", flush=True)
        if ao_medir and cada and (p + 1) % cada == 0:
            ao_medir(rede, p + 1)
    return rede


def mede(rede, amostras, tol_f1=None, guarda=None):
    """guarda: dicionário que recebe a densidade de cada trilha ('video|trilha' normal e 'video|trilha|ad' adaptada)."""
    out = []
    for s in amostras:
        d = densidade(rede, s)
        l = dict(grupo=s["grupo"], video=s["video"], trilha=int(s["trilha"]), estilo=int(s["estilo"]), real=float(s["total"]), rede=float(d.sum()))
        if guarda is not None:
            guarda[f"{s['video']}|{s['trilha']}"] = d.astype(np.float32)
            guarda[f"{s['video']}|{s['trilha']}|ad"] = densidade(rede, s, adapta=True).astype(np.float32)
            if getattr(rede, "cab_nado", None) is not None:
                guarda[f"{s['video']}|{s['trilha']}|nado"] = nadando(rede, s).astype(np.float32)
        if tol_f1 and s["ev"] is not None:
            l["f1"] = f1_posicao(d, s["ev"], tol_f1)
        out.append(l)
    return out


def resumo(linhas, nome):
    e = np.array([abs(l["rede"] - l["real"]) for l in linhas])
    r = np.array([l["real"] for l in linhas])
    txt = f"{nome}: {len(linhas)} nadadores | erro médio {e.mean():.2f} | ±1 {100 * (e <= 1).mean():.0f}% | ±2 {100 * (e <= 2).mean():.0f}% | soma {sum(l['rede'] for l in linhas):.0f}/{r.sum():.0f}"
    if linhas and "f1" in linhas[0]:
        a, p, n = (sum(l["f1"][k] for l in linhas if "f1" in l) for k in range(3))
        txt += f" | F1 de posição {2 * a / max(1, p + n):.3f}"
    return txt


if __name__ == "__main__":
    rotulo, passos, T, modo = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    op = dict(a.split("=") for a in sys.argv[5:])
    fora, semente, larg = op.get("fora"), int(op.get("semente", 0)), int(op.get("larg", 96))
    CFG["classe"], CFG["forte"] = op.get("classe", "Rede2"), op.get("aug") == "forte"
    if "vel" in op:
        CFG["vel"] = tuple(float(x) for x in op["vel"].split(","))
    CFG["nos"] = int(op.get("nos", 10))
    CFG["lr"] = float(op["lr"]) if "lr" in op else None
    CFG["pen"], CFG["nado"] = float(op.get("pen", 0)), float(op.get("nado", 0))
    SAI.mkdir(parents=True, exist_ok=True)
    assert op.get("fora_por", "prova") in ("prova", "arquivo"), "fora_por=prova (padrão) ou fora_por=arquivo"
    W, N = carrega(T, op.get("dados", "rede_dados_48.pkl"))
    if modo == "lovo":
        # validação deixando uma prova nossa de fora por vez: ajuste fino (a partir de 'inicial', que nunca viu vídeo nosso)
        # com a base Woinoski de treino + os nossos vídeos das outras provas; mede só nos vídeos da prova que ficou de fora.
        # Vídeos da mesma prova saem do treino juntos e o mesmo treino mede os dois; fora_por=arquivo desfaz isso.
        arq, arq_d = SAI / f"{rotulo}.json", SAI / f"{rotulo}_dens.npz"
        res = json.loads(arq.read_text()) if arq.exists() else {}
        dens = dict(np.load(arq_d)) if arq_d.exists() else {}
        Wtr = [s for s in W if s["grupo"] not in VAL_W]
        existem = {s["grupo"] for s in N}
        pedidos = op["grupos"].split(",") if "grupos" in op else sorted(existem)
        juntos = [set(x.split("+")) for x in op["juntos"].split(",")] if "juntos" in op else []
        assert not (juntos and op.get("fora_por") == "arquivo"), "juntos= não combina com fora_por=arquivo"
        errados = sorted(set(pedidos).union(*juntos) - existem)   # nome errado em juntos= deixaria no treino o que devia sair
        assert not errados, f"grupos que não existem nos dados: {errados} (existem: {sorted(existem)})"
        for g in pedidos:
            if g in res:
                continue
            t0 = time.time()
            fora_g = {g} if op.get("fora_por") == "arquivo" else fora_com(g, juntos)
            nome_fora = "+".join(sorted(fora_g))
            rede = treina(Wtr + [s for s in N if s["grupo"] not in fora_g], passos, T, larg, semente, inicial=(SAI / op["inicial"]) if "inicial" in op else None)
            for h in [g] + sorted(x for x in fora_g if x != g and x in pedidos and x not in res):
                torch.save(dict(estado=estado_cpu(rede), T=T, larg=larg, modo=modo, fora=nome_fora, passos=passos, **receita()), SAI / f"{rotulo}__{h}.pt")
                res[h] = mede(rede, [s for s in N if s["grupo"] == h and s["det"] == "m_final"], guarda=dens)
                for l in res[h]:
                    l["fora"] = nome_fora                  # o que ficou fora do treino do modelo que mediu este nadador
                np.savez_compressed(arq_d, **dens)
                arq.write_text(json.dumps(res))
                print("  " + resumo(res[h], h) + f" | {time.time() - t0:.0f} s | " + " ".join(f"{l['rede']:.1f}/{l['real']:.0f}" for l in res[h][:9])
                      + (f" | fora do treino: {nome_fora}" if len(fora_g) > 1 else ""), flush=True)
        provas = [l for g in res if g != "istock" for l in res[g]]
        print("  " + resumo(provas, "PROVAS (todas)"))
        sys.exit(0)
    if modo == "w":
        tr = [s for s in W if s["grupo"] not in VAL_W]
    elif modo == "wtudo":                              # toda a base Woinoski, nenhum vídeo nosso
        tr = list(W)
    elif modo == "w+n":                                # a prova inteira do vídeo 'fora' sai do treino; mede-se só no 'fora'
        fora_g = {fora} if (op.get("fora_por") == "arquivo" or not fora) else da_mesma_prova(fora)
        tr = W + [s for s in N if s["grupo"] not in fora_g]
    else:
        tr = W + N
    print(f"{rotulo}: treino com {len(tr)} trilhas ({sum(len(s['X']) for s in tr)} quadros), T={T}, {passos} passos", flush=True)
    rede = treina(tr, passos, T, larg, semente, log=True, inicial=(SAI / op["inicial"]) if "inicial" in op else None)
    torch.save(dict(estado=estado_cpu(rede), T=T, larg=larg, modo=modo, fora=fora, passos=passos, **receita()), SAI / f"{rotulo}.pt")
    res = {}
    if modo == "w":
        val = [s for s in W if s["grupo"] in VAL_W]
        res["woinoski_val"] = mede(rede, val, tol_f1=3 * 15 / 28)
        for k, n in enumerate(("crawl", "borboleta", "costas", "peito")):
            print("  " + resumo([l for l in res["woinoski_val"] if l["estilo"] == k], f"Woinoski validação {n}"))
    alvo = [s for s in N if s["det"] == "m_final" and (modo != "w+n" or s["grupo"] == fora)]
    res["nossos"] = mede(rede, alvo)
    for g in sorted({l["grupo"] for l in res["nossos"]}):
        ls = [l for l in res["nossos"] if l["grupo"] == g]
        print("  " + resumo(ls, g) + " | " + " ".join(f"{l['rede']:.1f}/{l['real']:.0f}" for l in ls[:9]))
    provas = [l for l in res["nossos"] if l["grupo"] != "istock"]
    if len(provas) > 1:
        print("  " + resumo(provas, "PROVAS (todas)"))
    (SAI / f"{rotulo}.json").write_text(json.dumps(res))
