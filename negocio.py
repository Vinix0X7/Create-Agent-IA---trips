"""Regras de negócio do Rota Certa: consultas sobre destinos, atrações, clima, câmbio, orçamento e pré-reserva.

Funções puras em Python. Na Etapa 2 elas viram ferramentas (@tool) do LangChain; na Etapa 4 o servidor MCP
expõe exatamente estas mesmas funções. Regra de erro (Aula 04):
- entrada inválida  -> levanta ValueError (no MCP vira isError=true);
- destino não cadastrado -> devolve {"encontrado": False, ...} (resultado de negócio, não é erro).
Dados didáticos em dados.json: preços, cotações e clima são aproximados.
"""
import json
import math
import random
import string
import unicodedata
from datetime import date, datetime
from pathlib import Path

PASTA = Path(__file__).resolve().parent
DADOS = json.loads((PASTA / "dados.json").read_text(encoding="utf-8"))
DIAS_SEMANA = ["segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo"]
CATEGORIAS = ["cultural", "gastronomico", "aventura", "relax", "familia", "compras", "natureza"]
HOSPEDAGEM = ["economica", "media", "superior"]
NOMES_MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro",
               "outubro", "novembro", "dezembro"]


def normalizar(texto) -> str:
    """Minúsculas, sem acentos e sem espaços extras: 'Tóquio ' -> 'toquio'."""
    t = unicodedata.normalize("NFKD", str(texto or "")).encode("ascii", "ignore").decode("ascii")
    return " ".join(t.lower().replace("_", " ").replace("-", " ").split())


def resolver_destino(nome) -> str | None:
    """Devolve o código do destino (ex.: 'lisboa') a partir do nome ou apelido, ou None."""
    alvo = normalizar(nome)
    for codigo, info in DADOS["destinos"].items():
        nomes = {normalizar(codigo), normalizar(info["nome"]), *[normalizar(a) for a in info["aliases"]]}
        if alvo in nomes:
            return codigo
    return None


def _texto_obrigatorio(valor, campo):
    if valor is None or not str(valor).strip():
        raise ValueError(f"O argumento '{campo}' é obrigatório e não pode ser vazio.")
    return str(valor).strip()


def _data(valor, campo) -> date:
    try:
        return datetime.strptime(str(valor).strip(), "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"O argumento '{campo}' deve estar no formato AAAA-MM-DD (recebido: {valor!r}).")


def para_brl(valor, moeda) -> float:
    return round(valor * DADOS["cotacoes_para_brl"][moeda], 2)


def _nao_encontrado(destino):
    return {"encontrado": False, "destino_pedido": destino,
            "mensagem": "Destino não cadastrado na base deste assistente.",
            "destinos_disponiveis": [d["nome"] for d in DADOS["destinos"].values()]}


def listar_destinos() -> dict:
    """Lista os destinos disponíveis, com país e moeda."""
    return {"total": len(DADOS["destinos"]),
            "destinos": [{"codigo": c, "nome": d["nome"], "pais": d["pais"], "moeda": d["moeda"], "aliases": d["aliases"]}
                         for c, d in DADOS["destinos"].items()]}


def buscar_atracoes(destino: str, categorias: list[str] | None = None, data: str | None = None,
                    somente_acessiveis: bool = False, somente_para_criancas: bool = False) -> dict:
    """Busca atrações de um destino, com preço em reais. Se 'data' (AAAA-MM-DD) vier, informa se abre nesse dia."""
    destino = _texto_obrigatorio(destino, "destino")
    codigo = resolver_destino(destino)
    if codigo is None:
        return _nao_encontrado(destino)
    cats = [normalizar(c) for c in (categorias or []) if str(c).strip()]
    invalidas = [c for c in cats if c not in CATEGORIAS]
    if invalidas:
        raise ValueError(f"Categorias inválidas: {invalidas}. Use: {CATEGORIAS}.")
    dia = DIAS_SEMANA[_data(data, "data").weekday()] if data else None
    info = DADOS["destinos"][codigo]
    atracoes = []
    for a in info["atracoes"]:
        if cats and not set(cats) & set(a["categorias"]):
            continue
        if (somente_acessiveis and not a["acessivel"]) or (somente_para_criancas and not a["bom_para_criancas"]):
            continue
        item = {**a, "preco_brl": para_brl(a["preco_local"], info["moeda"])}
        if dia:
            item["aberto_na_data"] = dia not in a["fechado_em"]
        atracoes.append(item)
    return {"encontrado": True, "destino": info["nome"], "dia_semana": dia, "total": len(atracoes), "atracoes": atracoes}


def consultar_clima(destino: str, mes: int) -> dict:
    """Médias climáticas aproximadas do destino no mês (1 a 12): temperaturas e dias de chuva."""
    destino = _texto_obrigatorio(destino, "destino")
    if not isinstance(mes, int) or not 1 <= mes <= 12:
        raise ValueError(f"O argumento 'mes' deve ser um inteiro de 1 a 12 (recebido: {mes!r}).")
    codigo = resolver_destino(destino)
    if codigo is None:
        return _nao_encontrado(destino)
    c = DADOS["destinos"][codigo]["clima"][str(mes)]
    chance = "alta" if c["dias_de_chuva"] >= 9 else "média" if c["dias_de_chuva"] >= 5 else "baixa"
    return {"encontrado": True, "destino": DADOS["destinos"][codigo]["nome"], "mes": NOMES_MESES[mes - 1],
            "temp_min_c": c["temp_min_c"], "temp_max_c": c["temp_max_c"], "dias_de_chuva": c["dias_de_chuva"],
            "chance_de_chuva": chance, "observacao": "Médias aproximadas, não é previsão do tempo."}


def converter_moeda(valor: float, de: str, para: str = "BRL") -> dict:
    """Converte valores entre BRL, EUR, USD e ARS com cotação didática fixa."""
    cot = DADOS["cotacoes_para_brl"]
    de, para = _texto_obrigatorio(de, "de").upper(), _texto_obrigatorio(para, "para").upper()
    moedas = [m for m in cot if not m.startswith("_")]
    if de not in moedas or para not in moedas:
        raise ValueError(f"Moeda não suportada. Use: {moedas}.")
    if valor is None or valor < 0:
        raise ValueError("O valor deve ser maior ou igual a zero.")
    return {"valor_original": valor, "de": de, "para": para,
            "valor_convertido": round(valor * cot[de] / cot[para], 2), "referencia": "cotação didática fixa"}


def calcular_orcamento(destino: str, noites: int, pessoas: int, atracao_ids: list[str] | None = None,
                       categoria_hospedagem: str = "media", incluir_passagem: bool = True) -> dict:
    """Custo estimado em reais: passagem (saindo de SP), hospedagem, alimentação, transporte local e atrações."""
    destino = _texto_obrigatorio(destino, "destino")
    if not isinstance(noites, int) or not 1 <= noites <= 30:
        raise ValueError("'noites' deve ser um inteiro entre 1 e 30.")
    if not isinstance(pessoas, int) or not 1 <= pessoas <= 10:
        raise ValueError("'pessoas' deve ser um inteiro entre 1 e 10.")
    if categoria_hospedagem not in HOSPEDAGEM:
        raise ValueError(f"categoria_hospedagem inválida. Use: {HOSPEDAGEM}.")
    codigo = resolver_destino(destino)
    if codigo is None:
        return _nao_encontrado(destino)
    info = DADOS["destinos"][codigo]
    dias, quartos = noites + 1, math.ceil(pessoas / 2)
    por_id = {a["id"]: a for a in info["atracoes"]}
    ids = [i.strip().upper() for i in (atracao_ids or [])]
    validos = [i for i in ids if i in por_id]
    itens = {
        "passagem_aerea": info["passagem_estimada_brl_por_pessoa"] * pessoas if incluir_passagem else 0,
        "hospedagem": info["hospedagem_noite_quarto_brl"][categoria_hospedagem] * noites * quartos,
        "alimentacao": info["alimentacao_dia_pessoa_brl"] * dias * pessoas,
        "transporte_local": info["transporte_local_dia_pessoa_brl"] * dias * pessoas,
        "atracoes": round(sum(para_brl(por_id[i]["preco_local"], info["moeda"]) for i in validos) * pessoas, 2),
    }
    return {"encontrado": True, "destino": info["nome"], "noites": noites, "pessoas": pessoas, "quartos": quartos,
            "categoria_hospedagem": categoria_hospedagem, "itens_brl": itens, "total_brl": round(sum(itens.values()), 2),
            "ids_ignorados": [i for i in ids if i not in por_id]}


def verificar_requisitos_entrada(destino: str, nacionalidade: str = "brasileira") -> dict:
    """Documentos e visto exigidos para entrar no destino (base atual: viajantes brasileiros)."""
    destino = _texto_obrigatorio(destino, "destino")
    codigo = resolver_destino(destino)
    if codigo is None:
        return _nao_encontrado(destino)
    nac = "brasileira" if normalizar(nacionalidade) in {"brasileira", "brasileiro", "brasil", "br"} else normalizar(nacionalidade)
    req = DADOS["destinos"][codigo]["requisitos_entrada"].get(nac)
    if req is None:
        return {"encontrado": False, "mensagem": f"Sem requisitos cadastrados para a nacionalidade '{nacionalidade}'."}
    return {"encontrado": True, "destino": DADOS["destinos"][codigo]["nome"], **req,
            "aviso": "Regras de entrada mudam. Confirme na fonte oficial antes de viajar."}


PRE_RESERVAS: list[dict] = []


def registrar_pre_reserva(destino: str, atracao_ids: list[str], data_inicio: str, pessoas: int,
                          confirmado_pelo_usuario: bool = False) -> dict:
    """Registra uma PRÉ-RESERVA sem pagamento. Ação COM EFEITO: só com confirmado_pelo_usuario=true."""
    if confirmado_pelo_usuario is not True:
        raise ValueError("Pré-reserva recusada: é necessária a confirmação explícita do usuário.")
    destino = _texto_obrigatorio(destino, "destino")
    if _data(data_inicio, "data_inicio") < date.today():
        raise ValueError("data_inicio não pode estar no passado.")
    codigo = resolver_destino(destino)
    if codigo is None:
        return _nao_encontrado(destino)
    por_id = {a["id"]: a for a in DADOS["destinos"][codigo]["atracoes"]}
    ids = [i.strip().upper() for i in (atracao_ids or [])]
    if not ids or any(i not in por_id for i in ids):
        raise ValueError("Informe ids de atrações existentes neste destino.")
    registro = {"codigo": "PRE-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6)),
                "destino": DADOS["destinos"][codigo]["nome"], "atracoes": [por_id[i]["nome"] for i in ids],
                "data_inicio": data_inicio, "pessoas": pessoas, "status": "pre_reserva_sem_pagamento"}
    PRE_RESERVAS.append(registro)
    return {"registrado": True, **registro}
