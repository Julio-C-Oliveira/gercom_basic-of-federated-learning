# Plano de Pesquisa: [TOPIC_TITLE]

**Assunto**: `[SUBJECT_NAME]` (`[SUBJECT_SLUG]`)  
**Tópico ID**: `[TOPIC_ID]` | **Fase**: `[TOPIC_PHASE]`  
**Domínio / Eixo Temático**: `[DOMÍNIO / SUBTEMA]`  
**Data**: [YYYY-MM-DD]  
**Status**: [Planejado / Em Execução / Concluído]  

---

## 1. Objetivo da Investigação & Metas de Aprendizado

[Descreva claramente o que deve ser compreendido e dominado ao final deste tópico. Qual competência conceitual e habilidade prática de implementação devem ser consolidadas?]

---

## 2. Perguntas-Chave de Investigação

As respostas a estas perguntas devem compor a espinha dorsal do documento de pesquisa:

1. **Fundamentos & Mecanismo Teórico**: Como o conceito/algoritmo opera matematicamente ou conceitualmente sob condições normais e quais são suas premissas centrais?
2. **Ambiente & Ferramental**: Quais bibliotecas, APIs, pacotes ou ferramentas são necessários e como configurá-los de forma limpa e reprodutível?
3. **Implementação & Pipeline Prático**: Qual o fluxo de código passo a passo para colocar a técnica/algoritmo em funcionamento com dados reais ou sintéticos?
4. **Desafios, Gargalos & Casos de Borda**: Quais as principais limitações conhecidas (ex.: dados heterogêneos/não-IID, overhead de comunicação, limitações de hardware, latência) e como tratá-las?
5. **Métricas de Avaliação & Diagnóstico**: Quais indicadores (acurácia, perda, taxa de convergência, throughput, uso de memória) medem o sucesso da solução?
6. **Boas Práticas & Estado da Arte**: Quais são os padrões recomendados pela comunidade, papers seminais ou referências canônicas sobre o tópico?

---

## 3. Bibliotecas, Frameworks & Recursos Necessários

| Ferramenta / Biblioteca | Propósito no Estudo | Fonte / Instalação |
| :--- | :--- | :--- |
| `[ferramenta-1]` | [ex.: Framework de simulação federada] | [pip install flwr / conda] |
| `[ferramenta-2]` | [ex.: Framework de deep learning base] | [PyTorch / Torchvision] |

---

## 4. Ambiente de Teste, Dataset & Cenário Prático

- **Ambiente Recomendado**: [Jupyter Notebook / Google Colab / Script Python local / Docker]
- **Dataset / Cenário**: [Nome do dataset ou conjunto de dados sintéticos para experimentação]
- **Instruções rápidas de setup**:
  ```bash
  # Comandos de setup do ambiente ou instalação de dependências
  pip install -r requirements.txt
  ```

---

## 5. Critérios de Aceite para Conclusão

Para que a pesquisa seja aprovada pelo comando `/verify`:

- [ ] Teoria e mecanismo conceitual/matemático descritos com precisão técnica.
- [ ] Pipeline e instruções de configuração/ambiente detalhados.
- [ ] Código ou implementação prática fornecida com **parâmetros, funções e hiperparâmetros comentados**.
- [ ] Análise clara de desafios técnicos, limitações e trade-offs.
- [ ] Métricas de avaliação e critérios de validação dos resultados documentados.
- [ ] Boas práticas, referências bibliográficas ou papers recomendados.
- [ ] Snippets, comandos ou parâmetros essenciais adicionados ao `cheatsheets/quick-reference.md`.
