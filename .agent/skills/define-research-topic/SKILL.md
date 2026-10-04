---
name: define-research-topic
description: Gera o plano de pesquisa investigativa detalhado para o próximo tópico pendente no roadmap do assunto ativo. Use este skill para planejar e estruturar perguntas-chave, ferramentas, datasets e critérios de aceite antes de iniciar a pesquisa de um tópico.
---

# Definir Tópico de Pesquisa

## User Input

```text
$ARGUMENTS
```

Você **DEVE** considerar o input do usuário (se fornecido). O input pode ser:
1. **Conhecimentos prévios, anotações ou foco desejado**: O que você já sabe sobre o assunto, suas dúvidas específicas ou bibliotecas que deseja enfatizar (ex.: `/define-research-topic Já domino PyTorch básico, quero focar na implementação do cliente e servidor no Flower framework usando FedAvg`).
2. **ID do tópico**: Para forçar um tópico específico (ex.: `/define-research-topic T02`).
3. **Assunto específico**: Caso deseje alternar o assunto ativo (ex.: `/define-research-topic 01-intro-federated-learning`).
4. **Vazio**: O comando selecionará automaticamente o primeiro tópico pendente (`[ ]`) do assunto ativo.

> **Importante**: Se você compartilhar o que já sabe ou suas notas pessoais, o plano será calibrado para **não perder tempo com o básico que você já domina**, focando as perguntas investigativas e os experimentos nas suas lacunas e em tópicos avançados.

---

## Scope Guard & Diretrizes

- O objetivo deste comando é **planejar a investigação**, formulando perguntas rigorosas, mapeando frameworks, datasets, experimentos e critérios de aceite.
- Este comando **NÃO** deve redigir a pesquisa completa nem marcar o tópico como concluído (`[x]`); a pesquisa cabe exclusivamente ao comando `/do-research`.
- Se o plano para aquele tópico já existir, informe o usuário e pergunte se ele deseja revisá-lo ou regenerá-lo.

---

## Fluxo de Execução

### 1. Resolução do Assunto Ativo

1. Se o usuário forneceu um slug de assunto em `$ARGUMENTS`, use-o.
2. Caso contrário, leia o arquivo de contexto `.agent/context.json` para obter `"active_subject"`.
3. Se `.agent/context.json` não existir ou estiver vazio:
   - Liste os diretórios em `subjects/`.
   - Se houver apenas 1, assuma-o automaticamente.
   - Se houver mais de um, liste as opções e solicite ao usuário qual assunto deseja continuar.

---

### 2. Leitura do Roadmap e Seleção do Tópico

1. Abra e leia `subjects/<subject-slug>/roadmap.md`.
2. Identifique o tópico-alvo:
   - Se o usuário especificou um ID (ex.: `T02`), localize a linha correspondente na Tabela de Controle.
   - Se não especificou, encontre o **primeiro tópico cujo status seja `[ ]` (pendente)**.
3. Se todos os tópicos já estiverem marcados como `[x]`:
   - Parabenize o usuário pela conclusão do roadmap!
   - Sugira a execução imediata do comando `/verify` para auditar a completude e consistência de todo o estudo.
   - Encerre a execução.

---

### 3. Geração do Plano Investigativo

1. Extraia os dados do tópico selecionado:
   - `ID` (ex.: `T01`, `T02`)
   - `Fase` (ex.: `01. Fundamentos Teóricos & Arquitetura`, `02. Frameworks, Setup & Ferramental`)
   - `Nome do Tópico`
2. Gere um slug para o tópico (ex.: `01-fundamentos-teoricos-fl`, `02-setup-flower-pytorch`).
3. Carregue a estrutura de [templates/plan-template.md](file:///home/julio/Documentos/gercom_basic-of-federated-learning/templates/plan-template.md) e preencha com alta especificidade técnica:
   - **Objetivo da Investigação**: Metas conceituais e práticas claras.
   - **4 a 6 Perguntas-Chave**: Questões profundas adaptadas ao tema (mecanismo conceitual/matemático, bibliotecas e configuração, fluxo de implementação, desafios/heterogeneidade e métricas de avaliação).
   - **Bibliotecas & Ferramental**: Lista de pacotes, frameworks e dependências (ex.: PyTorch, Flower, NumPy, SciPy, Scikit-learn, etc.).
   - **Ambiente & Datasets Recomendados**: Ambientes de experimentação (Jupyter, scripts locais, Docker) e datasets canônicos (ex.: MNIST, CIFAR-10, Shakespeare, dados sintéticos).
   - **Critérios de Aceite**: Checklist explícito dos requisitos que o documento de pesquisa deverá cumprir.
4. Salve o plano em:
   `subjects/<subject-slug>/plans/<id>-<topic-slug>-plan.md`

---

### 4. Atualização do `roadmap.md`

Atualize a linha do tópico na Tabela de Controle de `subjects/<subject-slug>/roadmap.md`:
- Na coluna **Plano Investigativo**, insira o link relativo para o plano recém-criado:
  `[Plano](plans/<id>-<topic-slug>-plan.md)`
- Atualize a data de "Última Atualização" no cabeçalho do roadmap.

---

### 5. Relatório de Conclusão para o Usuário

Apresente um resumo com:
- Tópico selecionado (`ID - Nome`).
- Perguntas investigativas formuladas no plano.
- Frameworks e dataset/ambiente recomendado.
- Caminho do arquivo gerado: `subjects/<subject-slug>/plans/<id>-<topic-slug>-plan.md`.
- Próximo passo recomendado: `Para conduzir a pesquisa técnica completa, execute: /do-research`
