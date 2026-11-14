"""Tela do Rota Certa em Streamlit. Publicação: GitHub + Streamlit Community Cloud (ver notebook, Etapa 6)."""
import asyncio
import os
import uuid

import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver

st.set_page_config(page_title="Rota Certa — Agente de Viagens", page_icon="✈️", layout="wide")

# A chave fica nos "Secrets" do Streamlit Cloud (nunca no código).
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

import agente  # noqa: E402  (importado depois de configurar a chave)

EXEMPLOS = {
    "Lisboa · casal · cultura": "Quero ir para Lisboa de 10 a 14 de março com minha esposa, orçamento de R$ 25 mil, gostamos de cultura e gastronomia",
    "Rio · família · orçamento apertado": "Quero levar minha família ao Rio de 20/12 a 23/12, somos 4 com 2 crianças, orçamento de 6 mil reais",
    "Pergunta: documentos": "Preciso de visto para ir a Buenos Aires?",
    "Falha: destino fora da base": "Quero ir para Tóquio",
}
PERIODOS = {"manha": "manhã", "tarde": "tarde", "noite": "noite", "dia inteiro": "dia inteiro"}
CORES = {"ok": "🟢", "nao_encontrado": "🟡", "erro": "🔴", "falha_conexao": "🔴"}


@st.cache_resource
def recursos():
    """Criados uma vez por servidor: modelo, base RAG e a memória das conversas (checkpointer)."""
    return agente.criar_modelo(), agente.criar_base(agente.PASTA / "guias"), InMemorySaver()


def enviar(texto: str):
    modelo, base, memoria = recursos()
    st.session_state.historico.append({"papel": "user", "texto": texto})
    with st.spinner("O agente está consultando o servidor MCP e os guias..."):
        try:
            estado = asyncio.run(agente.conversar(texto, st.session_state.thread_id, modelo, base, memoria))
        except Exception as erro:
            estado = {"resposta": f"Algo deu errado ({type(erro).__name__}). Tente de novo.", "caminho": [], "ferramentas": [], "fontes": []}
    st.session_state.historico.append({"papel": "assistant", "texto": estado.get("resposta", ""), "estado": estado})
    st.session_state.etapa = estado.get("etapa")


def mostrar_roteiro(r: dict):
    cor = "green" if r["dentro_do_orcamento"] else "red"
    st.markdown(f"**{r['destino']}** · {r['data_ida']} a {r['data_volta']} · {r['pessoas']} pessoa(s) · "
                f":{cor}[{agente.real(r['custo_total'])} de {agente.real(r['orcamento'])}]")
    for i, dia in enumerate(r["dias"], 1):
        linhas = [f"- *{PERIODOS.get(a['periodo'], a['periodo'])}* — {a['atividade']}" + (f" ({a['bairro']})" if a["bairro"] else "")
                  + (f" · {agente.real(a['custo_pessoa_brl'])}/pessoa" if a["custo_pessoa_brl"] else "") for a in dia["atividades"]]
        st.markdown(f"**Dia {i} · {dia['data']} ({dia['dia_semana']})**\n" + "\n".join(linhas))


def mostrar_decisao(e: dict):
    with st.expander("🔎 Como o agente decidiu"):
        st.markdown("**Caminho no grafo (LangGraph):** " + " → ".join(f"`{n}`" for n in e.get("caminho", [])))
        if e.get("ferramentas"):
            st.markdown("**Ferramentas chamadas**")
            st.dataframe([{"": CORES.get(f["status"], ""), "origem": f["origem"], "ferramenta": f["nome"],
                           "decidido por": f["decidido_por"], "status": f["status"], "argumentos": str(f["argumentos"]),
                           "erro": f.get("erro", "")} for f in e["ferramentas"]], hide_index=True, width="stretch")
        if e.get("fontes"):
            st.markdown("**Trechos dos guias (RAG)**")
            for f in e["fontes"]:
                st.caption(f"{f['documento']} · {f['secao']} · score {f['score']}: “{f['trecho'][:180]}…”")
        if e.get("historico_revisoes") and "revisar" in e.get("caminho", []):
            st.markdown("**Revisões do roteiro (Reflection)**")
            for h in e["historico_revisoes"]:
                st.caption(f"Tentativa {h['tentativa']}: {agente.real(h['total'])} — " + ("; ".join(h["problemas"]) or "aprovado"))
        for aviso in e.get("avisos", []):
            st.warning(aviso)
        if e.get("ferramentas_descobertas"):
            st.caption("Ferramentas descobertas no servidor MCP (tools/list): " + ", ".join(e["ferramentas_descobertas"]))


# ---------- estado da sessão do navegador
if "thread_id" not in st.session_state:
    st.session_state.thread_id, st.session_state.historico, st.session_state.etapa = str(uuid.uuid4()), [], None

with st.sidebar:
    st.title("✈️ Rota Certa")
    st.write("Agente de viagens com **LangGraph**, **RAG** e **servidor MCP**. Projeto Desafio — CAVI Senac 2026-2.")
    st.subheader("Cenários da demo")
    for rotulo, texto in EXEMPLOS.items():
        if st.button(rotulo, width="stretch"):
            enviar(texto)
    if st.button("🔄 Nova conversa", width="stretch"):
        st.session_state.thread_id, st.session_state.historico, st.session_state.etapa = str(uuid.uuid4()), [], None
    st.caption("Dados didáticos: preços, clima e regras são aproximados. Confirme nas fontes oficiais. "
               "Não envie CPF, passaporte ou cartão.")

st.header("Planeje sua viagem")
if not st.session_state.historico:
    st.info("Diga o destino, as datas, quantas pessoas e o orçamento. Destinos: Lisboa, Buenos Aires e Rio de Janeiro.")

for item in st.session_state.historico:
    with st.chat_message(item["papel"]):
        st.markdown(item["texto"])
        estado = item.get("estado")
        if estado:
            if estado.get("roteiro") and "apresentar" in estado.get("caminho", []):
                mostrar_roteiro(estado["roteiro"])
            mostrar_decisao(estado)

if st.session_state.etapa == "aguardando_confirmacao":
    st.warning("Há uma pré-reserva esperando sua decisão.")
    col1, col2 = st.columns(2)
    if col1.button("✅ Confirmar", width="stretch"):
        enviar("sim")
        st.rerun()
    if col2.button("❌ Cancelar", width="stretch"):
        enviar("cancelar")
        st.rerun()

if pergunta := st.chat_input("Ex.: Quero ir para Buenos Aires de 5 a 9 de abril, somos 3, até 13 mil"):
    enviar(pergunta)
    st.rerun()
