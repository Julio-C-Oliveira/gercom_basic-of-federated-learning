---
name: do-research
description: Executa a pesquisa técnica detalhada do tópico planejado, produzindo a documentação completa, cheatsheet rápido e atualizando o roadmap. Use este skill para conduzir a pesquisa técnica e científica em profundidade de um tópico planejado.
---

# Executar Pesquisa Técnica

## User Input

```text
$ARGUMENTS
```

Você **DEVE** considerar o input do usuário (se fornecido). Pode ser:
1. O ID do tópico a pesquisar (ex.: `T01`, `T02`).
2. O caminho direto para um arquivo de plano (ex.: `plans/T01-fundamentos-teoricos-fl-plan.md`).
3. Vazio: selecionará automaticamente o plano do primeiro tópico pendente com plano já gerado.

---

## Scope Guard & Diretrizes

- Este comando é responsável por **produzir o conhecimento técnico definitivo** e aprofundado do tópico.
- **Rigor e Profundidade**:
  - Evite respostas genéricas ou superficiais.
  - Todo comando de terminal ou chamada de função deve conter seus **parâmetros e flags explicados individualmente**.
  - Todo bloco de código deve conter comentários elucidativos e tratamento de fluxo.
  - A seção de **métricas de avaliação, diagnóstico e análise de trade-offs/limitações** é obrigatória.
- O comando deve **atualizar o status para `[x]` no `roadmap.md`** somente após o arquivo de pesquisa ser gravado com sucesso.
- O comando deve também **extrair snippets de código e parâmetros essenciais para o cheatsheet rápido** em `cheatsheets/quick-reference.md`.

---

## Fluxo de Execução

### 1. Resolução do Assunto e do Plano de Pesquisa

1. Obtenha o subject ativo via `.agent/context.json` ou pelo primeiro argumento.
2. Identifique o plano do tópico a ser executado:
   - Se o usuário passou um ID ou caminho de plano, use-o.
   - Caso contrário, leia `subjects/<subject-slug>/roadmap.md`:
     - Procure o primeiro tópico com status `[ ]` que já possua um link para `plans/...`.
3. Se nenhum plano for encontrado ou se o tópico selecionado não tiver plano gerado:
   - Avise o usuário: `"Nenhum plano encontrado para este tópico. Execute primeiro: /define-research-topic"`
   - Interrompa a execução.

---

### 2. Leitura e Análise do Plano

1. Abra e leia o arquivo de plano correspondente em `subjects/<subject-slug>/plans/<id>-<topic-slug>-plan.md`.
2. Mapeie todas as **Perguntas-Chave de Investigação**, as bibliotecas necessárias e os critérios de aceite estabelecidos.

---

### 3. Condução da Pesquisa Técnica em Profundidade

Responda exaustivamente aos requisitos do plano estruturando o conteúdo conforme o [templates/research-template.md](file:///home/julio/Documentos/gercom_basic_of_federated_learning/templates/research-template.md):

1. **Visão Geral & Teoria**:
   - Detalhe os fundamentos matemáticos, arquitetura e funcionamento interno do algoritmo ou conceito.
   - Explique a motivação da técnica e o impacto prático.
2. **Setup do Ambiente & Dependências**:
   - Forneça comandos limpos de instalação e configuração de pacotes.
   - Explique o propósito de cada biblioteca e versão recomendada.
3. **Implementação Prática & Algoritmo Passo a Passo**:
   - Apresente o código estruturado em ordem lógica de execução (ex.: definição de modelo, cliente, servidor, agregação).
   - Explique cada hiperparâmetro e função crítica.
4. **Desafios, Gargalos & Casos Extremos**:
   - Documente limitações conhecidas (ex.: dados não-IID, atraso de rede/stragglers, custo de memória).
   - Apresente estratégias recomendadas de mitigação.
5. **Experimento Prático / Prova de Conceito**:
   - Descreva um walkthrough com código ou script reproduzível com dados reais ou sintéticos.
   - Apresente saídas esperadas e evidências de execução saudável.
6. **Métricas de Avaliação & Diagnóstico**:
   - Apresente os indicadores de sucesso (acurácia, perda, taxa de convergência, uso de recursos).
   - Forneça exemplos de interpretação de logs.
7. **Boas Práticas & Referências**:
   - Aponte boas práticas de engenharia de software e referências científicas seminais (papers, documentação oficial).

Escreva o documento final em:
`subjects/<subject-slug>/research/<id>-<topic-slug>.md`

---

### 4. Atualização do Cheatsheet Rápido

Abra `subjects/<subject-slug>/cheatsheets/quick-reference.md`:
1. Extraia da pesquisa recém-concluída os snippets de código, comandos de execução e hiperparâmetros essenciais ("copiar e colar").
2. Adicione-os na seção apropriada de `quick-reference.md`.
3. Atualize a data de modificação no cabeçalho do cheatsheet.

---

### 5. Atualização do `roadmap.md`

Abra `subjects/<subject-slug>/roadmap.md`:
1. Altere o status do tópico de `[ ]` para `[x]`.
2. Adicione o link relativo na coluna **Documentação de Pesquisa**:
   `[Pesquisa](research/<id>-<topic-slug>.md)`
3. Recalcule as **Métricas de Progresso**:
   - Atualize a contagem de Concluídos e Pendentes.
   - Atualize a porcentagem: `Progresso = (Concluídos / Total) * 100%`.
4. Adicione uma linha no **Histórico de Sessões**:
   - `- [YYYY-MM-DD]: Tópico <ID> (<Nome>) concluído via /do-research.`
5. Atualize a data de "Última Atualização".

---

### 6. Relatório de Conclusão para o Usuário

Apresente um resumo com:
- Tópico concluído com sucesso (`ID - Nome`).
- Resumo dos principais conceitos, algoritmos e snippets documentados.
- Links para os arquivos gerados/atualizados:
  - `research/<id>-<topic-slug>.md`
  - `cheatsheets/quick-reference.md`
  - `roadmap.md` (com nova % de progresso).
- Próximo passo sugerido:
  - Se ainda houver tópicos pendentes: `Para planejar o próximo tópico, execute: /define-research-topic`
  - Se todos os tópicos foram concluídos: `Parabéns! Todos os tópicos foram finalizados. Execute /verify para auditar a completude dos seus estudos.`
