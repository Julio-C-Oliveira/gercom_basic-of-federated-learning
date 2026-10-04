---
name: define-subject
description: Inicializa um novo assunto ou módulo de estudo técnico, estruturando pastas em subjects/ e gerando o roadmap.md com metodologia científica e prática. Use este skill ao iniciar um novo tema ou importar conteúdos de estudo.
---

# Definir Assunto de Estudo

## User Input

```text
$ARGUMENTS
```

Você **DEVE** considerar o input do usuário antes de prosseguir. O input pode ser:
1. O caminho de um diretório `contents/` (ou `contents`) para **importar módulos em lote** (gerando os subjects de todos os arquivos encontrados).
2. O caminho de um arquivo específico em `contents/` (ex.: `contents/01_intro_federated_learning.md`).
3. Um nome de assunto em texto livre (ex.: `Federated Averaging (FedAvg)`, `Heterogeneidade de Dados e Não-IID`, `Privacidade Diferencial`).
4. Se vazio, pergunte ao usuário qual assunto ou tecnologia ele deseja estruturar para estudo ou se deseja importar arquivos de `contents/`.

---

## Scope Guard & Diretrizes

- Este comando inicializa o ambiente temático do(s) assunto(s) dentro da pasta `subjects/<subject-slug>/`.
- Ele **NÃO** deve gerar os planos detalhados ou pesquisas nesta etapa; apenas a estrutura de diretórios, o `roadmap.md` inicial e o arquivo de cheatsheet base para cada assunto.
- Sempre preserve arquivos pré-existentes caso o diretório do assunto já exista (não sobrescreva pesquisas ou planos existentes sem confirmação).

---

## Fluxo de Execução

### 1. Resolução do Assunto (Individual ou em Lote)

**Modo A: Importação em Lote da Pasta `contents/`**:
- Se o input for `contents/`, `contents` ou solicitar a importação de todos os arquivos de `contents`:
  1. Liste e ordene todos os arquivos Markdown presentes em `contents/`.
  2. Para **cada arquivo** na pasta:
     - Leia o conteúdo para extrair o objetivo, conceitos e ferramentas citadas.
     - Gere o slug padronizado mantendo o prefixo numérico para preservar a ordem pedagógica (ex.: `01-intro-federated-learning`).
     - Mapeie o domínio técnico e subtema correspondente.
     - Crie a pasta `subjects/<slug>/` com as subpastas `plans/`, `research/`, `cheatsheets/`.
     - Gere o `roadmap.md` específico daquele módulo a partir do template, analisando a profundidade do conteúdo para definir dinamicamente os tópicos necessários (`T01` a `TNN`), sem restrição a uma quantidade fixa.
     - Inicialize `cheatsheets/quick-reference.md`.
  3. Atualize o `README.md` raiz inserindo todos os módulos criados no Catálogo de Assuntos na ordem sequencial.
  4. Defina o primeiro módulo como o ativo no `.agent/context.json`.
  5. Pule para a etapa de Relatório de Conclusão informando todos os subjects criados.

**Modo B: Criação de Assunto Individual**:
- Se o input apontar para um arquivo específico em `contents/`:
  - Leia o arquivo para extrair título, objetivo, ferramentas e conceitos abordados.
  - Derive o slug (ex.: `01-federated-averaging`) e mapeie o subtema.
- Se for texto livre:
  - Extraia o nome principal do assunto e identifique o domínio técnico (Aprendizado Federado, Machine Learning, Redes, Otimização, etc.).
  - Gere o slug padronizado em minúsculas (ex.: `federated-averaging-fedavg`).

---

### 2. Criação da Estrutura de Pastas

Crie a estrutura de diretórios necessária para o assunto:

- `subjects/<subject-slug>/`
- `subjects/<subject-slug>/plans/`
- `subjects/<subject-slug>/research/`
- `subjects/<subject-slug>/cheatsheets/`

---

### 3. Elaboração do `roadmap.md` (Definição Dinâmica e Flexível de Tópicos)

Carregue a estrutura de [templates/roadmap-template.md](file:///home/julio/Documentos/gercom_basic_of_federated_learning/templates/roadmap-template.md).

**A quantidade de tópicos é flexível e adaptativa (NÃO há limite fixo de 5 tópicos)**:
- Analise a profundidade técnica, abrangência e complexidade do assunto em questão.
- Defina **quantos tópicos forem necessários** (`T01`, `T02`, ..., `TNN`) para cobrir o assunto com rigor e clareza, ultrapassando os 5 sempre que o assunto justificar (ex.: 3 a 4 tópicos para temas pontuais; 6, 8, 10 ou mais tópicos para assuntos densos).
- Se o usuário especificar no input a quantidade ou quais tópicos deseja, respeite fielmente.
- Cada tópico deve representar uma unidade coesa de estudo com escopo bem delimitado.
- Você pode utilizar ou desmembrar eixos temáticos conforme o assunto exigir, por exemplo:
  - **Fundamentos & Modelagem Teórica**: Mecanismos internos, formulação matemática e premissas.
  - **Frameworks, Setup & Ferramental**: Dependências, APIs e configuração do ambiente.
  - **Algoritmos & Implementação Prática**: Código principal, classes, funções e fluxo de execução.
  - **Variações & Casos Específicos**: Subestratégias, arquiteturas alternativas ou cenários especializados.
  - **Desafios, Heterogeneidade & Gargalos**: Dados não-IID, stragglers, restrições de hardware e overhead.
  - **Otimizações & Técnicas Avançadas**: Compressão de gradientes, privacidade diferencial, regularizações.
  - **Experimentos Práticos & Benchmarking**: Simulações com datasets canônicos e validação comparativa.
  - **Métricas, Avaliação & Boas Práticas**: Indicadores de convergência, diagnósticos e recomendações.

Escreva o arquivo preenchido com todos os tópicos planejados em:
`subjects/<subject-slug>/roadmap.md`

---

### 4. Inicialização do Cheatsheet Rápido

Crie o arquivo base para receber trechos de código e referências rápidas em:
`subjects/<subject-slug>/cheatsheets/quick-reference.md`

Com o seguinte formato inicial:
```markdown
# Cheatsheet Rápido: [SUBJECT_NAME]

**Assunto**: `[SUBJECT_SLUG]`  
**Última Atualização**: [YYYY-MM-DD]  

Arquivo de consulta rápida para comandos de ambiente, snippets de código essenciais e hiperparâmetros explorados durante o estudo.

---

## 1. Comandos de Ambiente & Dependências

*(Comandos de instalação, execução e setup alimentados automaticamente via `/do-research`)*

---

## 2. Snippets de Código & Configurações Essenciais

*(Trechos de código mais utilizados e chamadas de API alimentados via `/do-research`)*

---

## 3. Parâmetros, Hiperparâmetros & Dicas Práticas

*(Tabelas de hiperparâmetros e orientações práticas de experimentação)*
```

---

### 5. Atualização do Contexto Ativo

Para que os próximos comandos (`/define-research-topic`, `/do-research`, `/verify`) saibam automaticamente em qual assunto você está trabalhando, salve o contexto em `.agent/context.json`:

```json
{
  "active_subject": "<subject-slug>",
  "subject_name": "<subject-name>",
  "updated_at": "<TIMESTAMP_ISO>"
}
```

---

### 6. Atualização do Catálogo no README.md Principal

Se o `README.md` raiz possuir a seção de catálogo de assuntos, adicione ou atualize a linha da tabela com o novo assunto:
`| [<subject-name>](subjects/<subject-slug>/roadmap.md) | <dominio> | <subtema> | [ ] Em Progresso |`

---

### 7. Relatório de Conclusão para o Usuário

Apresente um resumo claro contendo:
- Nome e slug do assunto criado.
- Domínio técnico e subtema identificado.
- Tópicos definidos no roadmap.
- Caminhos absolutos dos arquivos criados.
- Próximo passo recomendado: `Você pode iniciar o planejamento do primeiro tópico digitando: /define-research-topic`
