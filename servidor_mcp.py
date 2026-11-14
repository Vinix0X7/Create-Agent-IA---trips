"""Servidor MCP "viagens" do Rota Certa (Aula 04).

Expõe as funções de negocio.py como ferramentas MCP. Roda como subprocesso via stdio:
o stdout é do protocolo (JSON-RPC), então este arquivo nunca usa print().
"""
import logging
import sys

from mcp.server.fastmcp import FastMCP

import negocio

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
mcp = FastMCP("viagens")

# Cada função de negocio.py vira uma ferramenta: o nome, a docstring e os tipos dos
# parâmetros formam o contrato (name, description, inputSchema) que o client descobre em tools/list.
for funcao in [negocio.listar_destinos, negocio.buscar_atracoes, negocio.consultar_clima, negocio.converter_moeda,
               negocio.calcular_orcamento, negocio.verificar_requisitos_entrada, negocio.registrar_pre_reserva]:
    mcp.tool()(funcao)


@mcp.resource("guia://{destino}")
def guia(destino: str) -> str:
    """Resumo de um destino: país, moeda e atrações."""
    codigo = negocio.resolver_destino(destino)
    if codigo is None:
        return f"Destino '{destino}' não cadastrado."
    info = negocio.DADOS["destinos"][codigo]
    return f"{info['nome']} ({info['pais']}), moeda {info['moeda']}. Atrações: " + ", ".join(a["nome"] for a in info["atracoes"])


if __name__ == "__main__":
    mcp.run(transport="stdio")
