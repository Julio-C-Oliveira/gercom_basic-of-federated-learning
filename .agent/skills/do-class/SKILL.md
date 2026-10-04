---
name: do-class
description: Conduz a elaboração de uma aula prática individual do cronograma da disciplina, apresentando previamente o assunto e o tempo sugerido para validação do usuário (aguardando 'Ok' ou ajustes), e redigindo a aula completa em Markdown segundo princípios pedagógicos de alto impacto.
---

# Elaborar Aula Prática da Disciplina

## User Input

```text
$ARGUMENTS
```

Você **DEVE** considerar o input do usuário. O input pode ser:
1. O número da aula (ex.: `01`, `Aula 02`).
2. Confirmação direta ou ajustes em resposta à proposta prévia (ex.: `Ok`, ou solicitações como *"reduza o tempo para 90 min e dê mais foco na customização da classe de cliente Flower"*).
3. Vazio: selecionará automaticamente a primeira aula pendente com status `[ ]` no cronograma da disciplina ativa.

---

## Scope Guard & Diretrizes

- Esta skill transforma tópicos de pesquisa de `subjects/` em **aulas práticas de excelência pedagógica**, salvas em `classes/<course-slug>/lessons/XX-<nome-da-aula>.md`.
- **Gate Obrigatório de Validação**: Você **NUNCA** deve gerar o arquivo da aula completa sem antes apresentar o assunto da aula e o tempo sugerido e obter o **"Ok"** explícito do usuário.
- **Títulos Limpos (Sem Metalinguagem Acadêmica)**: Não inclua parênteses metodológicos nos cabeçalhos da aula voltada ao aluno (evite rótulos como *"Regra de Feynman"*, *"Building a Fence - Winston"*, *"Ciclagem"* nos títulos). A aula deve soar 100% natural, técnica, didática e fluida.
- **Rigor Prático**: Toda aula deve conter código e comandos reais, com **parâmetros e funções explicados individualmente**, pipeline prático reproduzível e seção de troubleshooting de erros comuns.

---

## Fluxo de Execução

### 1. Resolução da Disciplina e da Aula Alvo

1. Identifique o curso ativo através de `.agent/class-context.json` (ou pelo primeiro argumento).
2. Se nenhum curso estiver ativo, avise o usuário:  
   `"Nenhuma disciplina ativa encontrada. Execute primeiro /define-class-roadmap para estruturar o curso."` e interrompa.
3. Abra e leia `classes/<course-slug>/class-roadmap.md`:
   - Se o usuário passou um número de aula específico, localize-a na tabela.
   - Caso contrário, selecione a **primeira aula com status `[ ]`**.
4. Leia o documento de pesquisa técnica correspondente em `subjects/<subject-slug>/research/` e o cheatsheet em `subjects/<subject-slug>/cheatsheets/quick-reference.md` para carregar a base conceitual e os snippets de código daquela aula.

---

### 2. Apresentação da Proposta & Gate de Validação (Obrigatório)

Antes de redigir a aula, apresente ao usuário um resumo executivo com a seguinte estrutura:

```markdown
### 📋 Proposta de Plano de Aula: Aula [XX] - [Título da Aula]

- **Disciplina**: [Nome do Curso]
- **Subject de Origem**: `subjects/[subject-slug]`
- **Tempo Sugerido**: [XX] minutos (ex.: 120 minutos)
- **Ambiente de Lab**: [Ex.: Jupyter Notebook / Python 3.10+ / Docker]

#### 🎯 Promessa desta Aula:
[3 competências práticas que o aluno saberá executar que não sabia no início]

#### 🗺️ Estrutura Proposta dos Marcos de Ensino:
1. **Intuição & Analogia do Problema**: [Conceito central e analogia do cotidiano].
2. **Delimitação de Escopo**: O que o conceito/algoritmo é vs. o que NÃO é.
3. **Execução Prática Hands-on**:
   - Marco 1: Preparação do ambiente e teste de dependências.
   - Marco 2: Mapeamento conceitual e preparação dos dados/modelo.
   - Marco 3: Implementação prática guiada e execução do pipeline com [Framework/Função].
   - Marco 4: Diagnóstico de resultados, métricas de avaliação e otimizações.
4. **Desafio Autônomo**: [Objetivo do exercício prático de fixação].

---
> ⚠️ **Aguardando sua confirmação**:
> Se você concordar com os assuntos da aula e com o tempo sugerido, responda **"Ok"**.
> Se desejar qualquer ajuste no tempo, foco prático ou bibliotecas abordadas, cite as alterações desejadas.
```

**Interrupção / Aguardo**:
- Se o usuário pedir alterações: reajuste o tempo, troque ou acrescente bibliotecas/tópicos e apresente a nova proposta.
- Apenas quando o usuário responder **"Ok"** (ou aprovação equivalente), prossiga para a Etapa 3.

---

### 3. Redação da Aula Completa em Markdown

Carregue [templates/class-lesson-template.md](file:///home/julio/Documentos/gercom_basic-of-federated-learning/templates/class-lesson-template.md) e elabore a aula completa aplicando rigorosamente os seguintes princípios:

1. **Abertura (Promessa Clara)**:
   - Inicie diretamente com o bloco `> [!IMPORTANT]` contendo o que o aluno saberá fazer ao final da sessão. Sem piadas de quebra de gelo ou conversas vazias.
2. **Intuição & Analogia do Problema**:
   - Introduza o conceito complexo com uma analogia intuitiva do mundo físico antes de entrar em detalhes matemáticos ou de código.
   - Explique por que o problema existe na prática e seu impacto real.
3. **O Que a Técnica É vs. O Que NÃO É**:
   - Construa uma tabela comparativa direta para eliminar equívocos comuns e delimitar o escopo.
4. **Roteiro Prático & Execução Passo a Passo (Marcos 1 a 4)**:
   - Divida o avanço em marcos explícitos (`### [Marco 1 de 4] ...`).
   - Código e comandos reais: **explique parâmetros e argumentos em lista comentada**.
   - Insira checkpoints intermediários para que alunos que pausaram saibam exatamente se estão no ponto certo (`> 🛑 Checkpoint: Suba de Volta no Ônibus`).
   - Insira perguntas socráticas reflexivas com a resposta técnica oculta em tags `<details><summary>💡 Clique para conferir a resposta técnica</summary>...</details>`.
   - Seção obrigatória de **Troubleshooting**: Liste pelo menos 2 erros comuns e como corrigi-los se o código ou comando falhar.
5. **Habilidades Práticas Conquistadas**:
   - Checklist com as capacidades operacionais dominadas durante a aula.
6. **Desafio Prático de Fixação**:
   - Missão autônoma com alvo claro e critério de sucesso objetivo (ex.: convergência mínima ou alteração de comportamento de cliente).
   - Sem despedidas frágeis com apenas "obrigado". Conecte diretamente com a próxima aula.

Salve o arquivo completo em:
`classes/<course-slug>/lessons/<XX>-<nome-da-aula-slug>.md`

---

### 4. Atualização do Roadmap da Disciplina

1. Abra `classes/<course-slug>/class-roadmap.md`.
2. Altere o status da aula de `[ ]` para `[x]`.
3. Atualize a coluna **Arquivo da Aula** inserindo o link relativo para o arquivo gerado:
   `[Acessar Aula](lessons/<XX>-<nome-da-aula-slug>.md)`
4. Recalcule e atualize a seção **Métricas de Execução**:
   - Horas Concluídas e Porcentagem.
   - Aulas Concluídas e Aulas Pendentes.
5. Adicione o registro no **Histórico de Modificações**.

---

### 5. Relatório de Conclusão

Apresente um resumo final ao usuário contendo:
- Caminho da aula gerada: `classes/<course-slug>/lessons/<XX>-<nome-da-aula-slug>.md`.
- Tempo estimado e habilidades cobertas.
- Status atualizado do progresso da disciplina (% de conclusão).
- Próxima aula na fila do cronograma.
