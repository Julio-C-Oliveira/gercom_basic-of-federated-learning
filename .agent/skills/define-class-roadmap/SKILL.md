---
name: define-class-roadmap
description: Analisa os subjects disponíveis no projeto, consulta os temas desejados e a carga horária total da matéria, calcula a distribuição de tempo por assunto e gera o cronograma da disciplina em classes/<course-slug>/class-roadmap.md.
---

# Definir Cronograma de Aulas da Matéria

## User Input

```text
$ARGUMENTS
```

Você **DEVE** considerar o input do usuário. O input pode conter:
1. Argumentos com o nome do curso e parâmetros iniciais (ex.: `aprendizado-federado-pratico --hours 40`).
2. Se vazio ou incompleto, conduza o fluxo interativo passo a passo com o usuário.

---

## Scope Guard & Diretrizes

- Esta skill é responsável pelo **planejamento curricular da matéria/disciplina**, transformando a base de pesquisa técnica existente em `subjects/` em uma grade pedagógica de aulas.
- A skill **NÃO** redige as aulas completas nesta etapa; seu papel é definir a distribuição temporal equilibrada, o escopo de cada aula e gerar o arquivo mestre `classes/<course-slug>/class-roadmap.md`.
- O cálculo de tempo deve levar em consideração a complexidade técnica dos temas: tópicos mais densos (ex.: Heterogeneidade não-IID, Algoritmos de Agregação Avançados, Privacidade Diferencial) demandam mais horas e subdivisão em mais aulas do que tópicos introdutórios.

---

## Fluxo de Execução

### 1. Varredura dos Subjects Disponíveis

1. Liste todas as pastas em `subjects/` (ignorando `.gitkeep` e arquivos avulsos).
2. Para cada subject encontrado:
   - Leia o cabeçalho de `subjects/<subject-slug>/roadmap.md` para extrair o nome formal do assunto e o progresso técnico.
3. Formate e apresente ao usuário o catálogo completo de assuntos disponíveis no projeto:
   ```text
   Assuntos disponíveis para composição da matéria:
   [01] 01-intro-federated-learning (Fundamentos de FL e Arquitetura)
   [02] 02-flower-framework-setup (Setup do Framework Flower e PyTorch)
   [03] 03-federated-averaging-fedavg (Algoritmo FedAvg e Agregação)
   [04] 04-non-iid-data-heterogeneity (Heterogeneidade Estatística)
   [05] 05-differential-privacy-fl (Privacidade Diferencial)
   ```

---

### 2. Coleta de Requisitos com o Usuário

Pergunte ou confirme com o usuário os seguintes parâmetros:

1. **Subjects Selecionados**: Quais assuntos farão parte desta matéria (ex.: "todos", "apenas 01, 02 e 03", etc.).
2. **Carga Horária Total**: Quantas horas totais a disciplina terá (ex.: 20h, 40h, 60h).
3. **Formato dos Encontros**: Duração média de cada aula (ex.: 2 horas / 120 min, 3 horas ou 4 horas).
4. **Nome da Disciplina**: Um título ou identificador para o curso (ex.: `Aprendizado Federado Prático` -> slug `aprendizado-federado-pratico`).

---

### 3. Cálculo & Sugestão de Distribuição Temporal

Com os dados em mãos, execute a distribuição de tempo:

1. **Ponderação por Complexidade Técnica**:
   - *Fundamentos & Setup (ex.: Conceitos Básicos, Ambiente Python)*: Peso 1x.
   - *Frameworks & Pipelines (ex.: Flower, PyTorch, Carga de Dados)*: Peso 1.5x.
   - *Algoritmos Principais & Treinamento (ex.: FedAvg, Estratégias Customizadas)*: Peso 2x.
   - *Tópicos Avançados & Otimização (ex.: Heterogeneidade não-IID, FedProx, Privacidade)*: Peso 2.5x a 3x.
2. **Cálculo da Quantidade de Aulas**:
   $$\text{Total de Aulas} = \frac{\text{Carga Horária Total}}{\text{Duração por Aula}}$$
3. **Distribuição dos Temas em Aulas Específicas**:
   - Se um subject complexo recebeu 6 horas e as aulas são de 2h, quebre o subject em 3 aulas focadas e complementares (ex.: Aula A: Intuição Teórica e Setup; Aula B: Implementação Guiada; Aula C: Otimizações e Casos Reais).
4. **Apresentação da Proposta ao Usuário**:
   - Apresente a grade sugerida com a tabela de aulas planejadas, tempo de cada uma e tópicos abordados para validação do usuário.

---

### 4. Geração do Cronograma da Disciplina

Após a confirmação da grade pelo usuário:

1. Crie a pasta `classes/<course-slug>/` e a subpasta `classes/<course-slug>/lessons/`.
2. Carregue o modelo canônico de [templates/class-roadmap-template.md](file:///home/julio/Documentos/gercom_basic-of-federated-learning/templates/class-roadmap-template.md).
3. Preencha todos os campos do template:
   - Slug, Carga horária total, formato de aulas e quantidade de aulas.
   - Tabela de subjects integrados com a carga horária alocada a cada um.
   - Tabela de controle de aulas com as aulas numeradas sequencialmente (`01`, `02`, ...), título, subject base, tempo em minutos e status `[ ]`.
4. Salve o arquivo em:
   `classes/<course-slug>/class-roadmap.md`
5. Registre o curso ativo no arquivo de contexto do agente:
   `.agent/class-context.json`
   ```json
   {
     "active_course": "<course-slug>",
     "course_name": "<Nome do Curso>",
     "total_hours": 40,
     "lesson_duration_minutes": 120,
     "roadmap_path": "classes/<course-slug>/class-roadmap.md",
     "updated_at": "YYYY-MM-DDTHH:MM:SS"
   }
   ```

---

### 5. Relatório de Conclusão

Apresente um resumo claro para o usuário:
- Caminho do roadmap gerado: `classes/<course-slug>/class-roadmap.md`.
- Total de aulas planejadas e carga horária consolidada.
- Instrução de próximo passo:  
  `"O cronograma da disciplina foi estruturado com sucesso! Para iniciar a elaboração da primeira aula, execute: /do-class"`
