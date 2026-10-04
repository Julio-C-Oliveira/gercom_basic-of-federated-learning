# Cheatsheet Rápido: Aprendizado Federado

**Assunto**: `aprendizado-federado`  
**Última Atualização**: 2026-10-04  

Arquivo de consulta rápida para comandos de ambiente, snippets de código essenciais e hiperparâmetros explorados durante o estudo de Redes Neurais (PyTorch) e Aprendizado Federado (Flower & NVFlare).

---

## 1. Comandos de Ambiente & Dependências

*(Comandos de instalação, execução e setup alimentados automaticamente via `/do-research`)*

### Setup Básico do Ambiente Virtual

```bash
# Criação e ativação do ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalação das dependências principais (PyTorch, Flower, NVFlare, Ray)
pip install torch torchvision torchaudio
pip install "flwr[simulation]>=1.8.0" flwr-datasets
pip install nvflare
```

---

## 2. Snippets de Código & Configurações Essenciais

*(Trechos de código mais utilizados e chamadas de API alimentados via `/do-research`)*

---

## 3. Parâmetros, Hiperparâmetros & Dicas Práticas

*(Tabelas de hiperparâmetros e orientações práticas de experimentação)*
