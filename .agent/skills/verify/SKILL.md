---
name: verify
description: Audita a integridade, links, consistência e profundidade técnica dos estudos concluídos no subject ativo. Use este skill para auditar a completude e consistência da base de conhecimento de um assunto.
---

# Auditoria e Verificação de Integridade

## User Input

```text
$ARGUMENTS
```

Você **DEVE** considerar o input do usuário (se fornecido). Pode ser:
1. O slug de um assunto específico para auditar (ex.: `01-intro-federated-learning`).
2. Vazio: auditará o assunto ativo definido em `.agent/context.json`.

---

## Scope Guard & Diretrizes

- **Somente Leitura (Read-Only)**: Este comando não deve alterar, criar ou deletar arquivos de documentação ou planos.
- O propósito exclusivo do comando é inspecionar e relatar a consistência da base de conhecimento, apontando lacunas ou inconsistências entre o `roadmap.md`, os planos em `plans/` e as pesquisas em `research/`.

---

## Fluxo de Execução

### 1. Resolução do Assunto Ativo

1. Obtenha o subject ativo via argumento ou leia `.agent/context.json`.
2. Se nenhum assunto for especificado ou ativo, liste os assuntos disponíveis em `subjects/` e solicite ao usuário qual deseja auditar.
3. Valide se a pasta `subjects/<subject-slug>/` e o arquivo `roadmap.md` existem. Se não existirem, aborte com mensagem de erro clara.

---

### 2. Varredura Cruzada de Integridade (Cross-Artifact Audit)

Execute a inspeção em 4 dimensões de qualidade:

#### Dimensão A: Integridade do Roadmap vs Arquivos Físicos
- Para cada tópico com status `[x]` no `roadmap.md`:
  - O arquivo apontado na coluna **Documentação de Pesquisa** existe fisicamente em `subjects/<subject-slug>/research/`?
  - O arquivo não está vazio (tamanho > 100 bytes)?
- Para cada link na coluna **Plano Investigativo**:
  - O arquivo apontado existe fisicamente em `subjects/<subject-slug>/plans/`?

#### Dimensão B: Profundidade Técnica Mínima da Pesquisa
Para cada arquivo em `research/*.md`, verifique se as seções essenciais estão presentes e com conteúdo substancial:
- [ ] **Teoria / Fundamentos**: Existe explicação conceitual ou formulação matemática?
- [ ] **Setup & Ferramental**: Há instruções claras de bibliotecas, dependências e ambiente?
- [ ] **Implementação Prática**: Há código estruturado com parâmetros e funções comentados?
- [ ] **Desafios & Limitações**: Há discussão de gargalos técnicos, heterogeneidade ou trade-offs?
- [ ] **Métricas & Avaliação**: Há definição de métricas de sucesso, logs ou diagnóstico de execução?

#### Dimensão C: Auditoria de Arquivos Órfãos
- Varra os arquivos físicos dentro de `subjects/<subject-slug>/plans/` e `subjects/<subject-slug>/research/`.
- Identifique se há arquivos existentes no disco que **NÃO constam na tabela do `roadmap.md`**.

#### Dimensão D: Estado do Cheatsheet Rápido
- Verifique se `subjects/<subject-slug>/cheatsheets/quick-reference.md` existe e se possui comandos/snippets adicionados para os tópicos que já foram marcados como concluídos (`[x]`).

---

### 3. Geração do Relatório de Auditoria

Apresente o resultado em uma tabela de conformidade Markdown com métricas claras:

```markdown
# Relatório de Auditoria: [SUBJECT_NAME]

**Assunto**: `[SUBJECT_SLUG]`  
**Data da Auditoria**: [YYYY-MM-DD HH:MM]  
**Status da Auditoria**: [✓ APROVADO / ⚠ PENDÊNCIAS ENCONTRADAS]  

---

## 1. Tabela de Conformidade dos Tópicos

| ID | Tópico | Status Roadmap | Arquivo de Pesquisa | Profundidade Técnica | Integridade |
| :--- | :--- | :---: | :---: | :---: | :---: |
| T01 | Fundamentos Teóricos | [x] | ✓ Presente | ✓ Completo (5/5) | ✓ OK |
| T02 | Setup & Frameworks | [x] | ✓ Presente | ⚠ Sem Métricas (4/5) | ⚠ Revisar |
| T03 | Algoritmos & Hands-on | [ ] | - | - | - |

---

## 2. Métricas de Integridade

- **Tópicos no Roadmap**: [Total]
- **Tópicos Concluídos (`[x]`)**: [N]
- **Documentos de Pesquisa Verificados**: [N]
- **Planos Verificados**: [N]
- **Arquivos Órfãos Detectados**: [0 ou lista de arquivos]
- **Cheatsheet Atualizado**: [Sim / Não / Parcial]

---

## 3. Apontamentos e Recomendações

[Liste qualquer problema encontrado, como seções ausentes, arquivos órfãos, links quebrados ou funções sem explicação de parâmetros.]

---

## 4. Próximos Passos Recomendados

- [Se houver pendências]: "Recomenda-se complementar a seção X do tópico Y."
- [Se tudo estiver correto e houver tópicos pendentes]: "Execute `/define-research-topic` para avançar ao próximo tópico."
- [Se o roadmap estiver 100% concluído e verificado]: "🎉 O assunto está integralmente pesquisado, auditado e pronto para uso e consulta!"
```
