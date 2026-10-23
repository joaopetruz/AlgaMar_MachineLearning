"""Dashboard AlgaMar no seu computador, sem depender do backend da outra máquina (100.109.64.40:3000).

Este servidor faz o papel do backend só para teste local:
- entrega as páginas do frontend (pasta Algamar_Page_Software), sem alterar nenhum arquivo dela;
- troca, só na entrega, o endereço da API do software/js/api-config.js por este servidor;
- responde /api/health, /api/auth/* (login de teste: qualquer e-mail e senha entram) e
  /api/ml/coastal-risks, que vem direto do ML (algamar_ml/api_dashboard.py, a previsão oficial).

Uso (venv do ML):
  python dashboard_local.py                      # API em http://localhost:3000/api + páginas em http://localhost:3000
  python dashboard_local.py --frontend OUTRA_PASTA --porta 3000

Duas formas de abrir o dashboard:
- pelo próprio servidor: http://localhost:3000/software/pages/login.html (não precisa mexer em nada);
- pelo Live Server do VS Code (porta 5501): em software/js/api-config.js, deixe ativa a linha
  window.ALGAMAR_API_BASE = "http://localhost:3000/api" (o CORS daqui libera a chamada).

Contas e perfis ficam só na memória (somem ao reiniciar). NÃO é o backend de produção.
"""
import argparse
import secrets
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import APIRouter, Body, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from algamar_ml.api_dashboard import monta

FRONTEND_PADRAO = Path.home() / "OneDrive" / "Desktop" / "Algamar_Page_Software"

CONFIG_LOCAL = """// Gerado por dashboard_local.py (teste local): a API é este mesmo servidor.
window.ALGAMAR_API_BASE = window.location.origin + "/api";
window.ALGAMAR_TOKEN_KEY = "algamar_token";
window.ALGAMAR_USER_KEY = "algamar_user";
"""

api = APIRouter(prefix="/api")
SESSOES = {}   # token → usuário


def _usuario(email, nome=None):
    nome = nome or email.split("@")[0]
    return {"id": email, "email": email, "fullName": nome, "full_name": nome}


def _nova_sessao(usuario):
    token = "local-" + secrets.token_hex(16)
    SESSOES[token] = usuario
    return {"success": True, "token": token, "user": usuario}


def _sessao(authorization):
    token = (authorization or "").removeprefix("Bearer ").strip()
    if token not in SESSOES and token != "demo-mode":
        raise HTTPException(401, "Sessão expirada.")
    return token


@api.get("/health")
def health():
    return {"success": True, "system": "Algamar (local)", "status": "online", "database": {"status": "local"}}


@api.post("/auth/login")
def login(corpo: dict = Body(...)):
    if not corpo.get("email") or not corpo.get("password"):
        raise HTTPException(400, "Informe e-mail e senha.")
    return _nova_sessao(_usuario(corpo["email"]))


@api.post("/auth/register")
def registrar(corpo: dict = Body(...)):
    if not corpo.get("email") or not corpo.get("password"):
        raise HTTPException(400, "Informe e-mail e senha.")
    return _nova_sessao(_usuario(corpo["email"], corpo.get("full_name")))


@api.put("/auth/me")
def atualizar(corpo: dict = Body(...), authorization: str | None = Header(None)):
    token = _sessao(authorization)
    u = dict(SESSOES.get(token) or _usuario("demo@algamar.local", "Usuário demonstração"))
    if corpo.get("email"):
        u.update(email=corpo["email"], id=corpo["email"])
    if corpo.get("full_name"):
        u.update(fullName=corpo["full_name"], full_name=corpo["full_name"])
    SESSOES[token] = u
    return {"success": True, "user": u}


@api.delete("/auth/me")
def excluir(authorization: str | None = Header(None)):
    SESSOES.pop(_sessao(authorization), None)
    return {"success": True}


@api.get("/ml/coastal-risks")
def coastal_risks(region: str | None = Query(None), risk_level: str | None = Query(None), month: int | None = Query(None),
                  authorization: str | None = Header(None)):
    _sessao(authorization)
    return monta(region, risk_level, month)


def cria_app(frontend: Path):
    if not (frontend / "software" / "js" / "api-config.js").exists():
        raise SystemExit(f"Não achei o frontend em {frontend} (falta software/js/api-config.js). Use --frontend PASTA.")
    app = FastAPI(title="AlgaMar: dashboard local")
    # libera o frontend aberto por outro servidor local (Live Server na 5501 etc.)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(HTTPException)
    def erro(_, exc):   # o frontend lê data.message
        return JSONResponse({"success": False, "message": exc.detail}, status_code=exc.status_code)

    app.include_router(api)

    @app.get("/software/js/api-config.js", include_in_schema=False)
    def config_js():
        return Response(CONFIG_LOCAL, media_type="application/javascript", headers={"Cache-Control": "no-store"})

    app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--frontend", type=Path, default=FRONTEND_PADRAO)
    ap.add_argument("--porta", type=int, default=3000)
    ap.add_argument("--sem-navegador", action="store_true")
    a = ap.parse_args()
    app = cria_app(a.frontend.resolve())
    url = f"http://localhost:{a.porta}/software/pages/login.html"
    print(f"API local: http://localhost:{a.porta}/api   Dashboard: {url}")
    print(f"Qualquer e-mail e senha entram. Pelo Live Server: use http://localhost:{a.porta}/api no api-config.js. Ctrl+C para parar.")
    if not a.sem_navegador:
        webbrowser.open(url)
    uvicorn.run(app, host="127.0.0.1", port=a.porta)
