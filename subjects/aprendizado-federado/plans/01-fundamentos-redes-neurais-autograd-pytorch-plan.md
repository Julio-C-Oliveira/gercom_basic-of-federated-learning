# Plano de Pesquisa: Fundamentos de Redes Neurais e Autograd no PyTorch

**Assunto**: `Aprendizado Federado` (`aprendizado-federado`)  
**Tópico ID**: `T01` | **Fase**: `01. Nivelamento PyTorch`  
**Domínio / Eixo Temático**: Deep Learning com PyTorch / Diferenciação Automática e Gestão de Tensores  
**Data**: 2026-10-04  
**Status**: Planejado  

---

## 1. Objetivo da Investigação & Metas de Aprendizado

O objetivo deste tópico é consolidar um entendimento rigoroso, conceitual e prático da espinha dorsal do PyTorch: a representação de dados via tensores (`torch.Tensor`) e o motor de diferenciação automática (`torch.autograd`). Ao final desta investigação, o pesquisador deverá:

1. Dominar a anatomia interna de um `torch.Tensor`, compreendendo layout de memória (armazenamento unidimensional, *strides*, contiguidade) e manipulação entre dispositivos (`cpu`, `cuda`, `mps`).
2. Desvendar o funcionamento do motor `torch.autograd`, especificamente o paradigma de rastreamento dinâmico baseado em fita (*tape-based dynamic autograd*), a construção do grafo acíclico direcionado (DAG) no *forward pass* e sua destruição no *backward pass*.
3. Compreender a mecânica exata da retropropagação via `loss.backward()`, o cálculo de produtos vetor-Jacobiano (VJP), o acúmulo padrão de gradientes (`tensor.grad`) e as técnicas de isolamento de tensores (`detach()`, `torch.no_grad()`, `torch.inference_mode()`).
4. Desenvolver autonomia para manipular diretamente tensores e gradientes sem auxílio de abstrações de alto nível, pavimentando o conhecimento indispensável para a extração, serialização, perturbação e agregação de pesos em ecossistemas de Aprendizado Federado (Flower e NVFlare).

---

## 2. Perguntas-Chave de Investigação

As respostas a estas perguntas constituirão o núcleo do documento de pesquisa:

1. **Tensores, Memória e Dispositivos**:
   - Como o PyTorch aloca e gerencia a memória de tensores (Storage vs. Tensor Head, *strides* e contiguidade)?
   - Qual é o custo computacional e de barramento na migração de tensores entre CPU e GPU (`.to(device)` / `.cuda()`), e como flags como `pin_memory=True` e `non_blocking=True` otimizam esse tráfego de dados?
2. **Mecanismo do Autograd e Grafo Computacional Dinâmico**:
   - Como o PyTorch rastreia operações em tempo de execução para gerar o grafo acíclico direcionado (DAG)?
   - O que diferencia um nó folha (`leaf tensor`) de nós intermediários, e qual o papel dos atributos `requires_grad`, `grad_fn` e `is_leaf`?
   - O que ocorre com o grafo computacional após a execução de `loss.backward()` e quando é necessário invocar `retain_graph=True`?
3. **Retropropagação e Acúmulo de Gradientes**:
   - Como funciona a matemática por trás de `loss.backward()` (diferenciação em modo reverso e produtos vetor-Jacobiano)?
   - Por que o PyTorch acumula gradientes em `tensor.grad` em vez de sobrescrevê-los, e como essa característica viabiliza *Gradient Accumulation* em dispositivos clientes com restrição severa de VRAM?
   - Qual a diferença prática e de consumo de memória entre `optimizer.zero_grad()` e `optimizer.zero_grad(set_to_none=True)`?
4. **Isolamento e Controle de Rastreamento**:
   - Quais as distinções exatas de performance, retenção de metadados e versionamento entre `torch.no_grad()`, `torch.inference_mode()`, `.detach()` e a modificação de `requires_grad_(False)`?
   - Em quais etapas do ciclo de vida de um nó federado (inferência local, extração de pesos para broadcast e cálculo de métricas de validação) cada mecanismo deve ser rigorosamente empregado?
5. **Inspeção, Hooks e Manipulação Manual de Gradientes**:
   - Como inspecionar e alterar gradientes programaticamente antes da etapa de otimização?
   - Como funcionam os hooks de tensores (`register_hook`) e como eles podem ser explorados para técnicas como *Gradient Clipping* manual, injeção de ruído gaussiano (Privacidade Diferencial) ou detecção de gradientes anômalos (*poisoning*)?
6. **Armadilhas de Memória e Implicações para Aprendizado Federado**:
   - Quais são os principais vetores de vazamento de memória (*memory leaks*) ao lidar com tensores e grafos (ex.: acumular perda retendo o grafo via `total_loss += loss` sem `.item()`)?
   - Como o gerenciamento deficiente de tensores e falta de descarte explícito de referências impacta simulações federadas de larga escala rodando centenas de clientes em uma única GPU?

---

## 3. Bibliotecas, Frameworks & Recursos Necessários

| Ferramenta / Biblioteca | Versão Mínima | Propósito no Estudo | Fonte / Instalação |
| :--- | :--- | :--- | :--- |
| `torch` | `>=2.1.0` | Manipulação de tensores multidimensionais, autograd e aceleração CUDA | `pip install torch` |
| `torchvision` | `>=0.16.0` | Utilitários de visão computacional e datasets canônicos de teste | `pip install torchvision` |
| `numpy` | `>=1.24.0` | Conversão e validação bidirecional (ponte entre tensores PyTorch e arrays serializados para FL) | `pip install numpy` |
| `torchviz` | `>=0.0.2` | Visualização gráfica do DAG gerado pelo autograd (exportação Graphviz) | `pip install torchviz` |
| `matplotlib` | `>=3.8.0` | Plotagem analítica de gradientes, perdas e verificação visual | `pip install matplotlib` |

---

## 4. Ambiente de Teste, Dataset & Cenário Prático

- **Ambiente Recomendado**: Script Python standalone (`.py`) executado em terminal Linux ou ambiente interativo Jupyter/VS Code, com suporte a GPU CUDA (ou CPU fallback).
- **Cenário Prático 1 (Inspeção Analítica do Grafo)**:
  - Construção de um DAG não-linear sintético com operações encadeadas ($\hat{y} = w_2 \cdot \sigma(w_1 x + b_1) + b_2$).
  - Inspeção manual dos nós `grad_fn`, verificação analítica da regra da cadeia vs. valores calculados por `backward()`, e visualização com `torchviz`.
- **Cenário Prático 2 (Loop de Otimização Manual)**:
  - Implementação de um classificador binário ou regressor linear sem utilizar `torch.optim` ou `nn.Module`.
  - Execução explícita do passo de atualização:
    ```python
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()
    ```
  - Comparação com o ciclo nativo do PyTorch (`optimizer.step()`), demonstrando como pesos são atualizados em um cliente federado antes do envio ao servidor agregador.
- **Instruções rápidas de setup**:
  ```bash
  pip install torch torchvision numpy torchviz matplotlib
  ```

---

## 5. Critérios de Aceite para Conclusão

Para que a documentação de pesquisa deste tópico seja aprovada pelo comando `/verify`:

- [ ] **Rigor Conceitual**: Descrição detalhada da estrutura interna do `torch.Tensor` (Storage, offset, strides, contiguidade) e do funcionamento do motor `torch.autograd`.
- [ ] **Diagramação Técnica**: Diagrama Mermaid ilustrando o ciclo de vida do DAG dinâmico (Forward pass -> Grafo em Memória -> Backward pass -> Acúmulo em `.grad` -> Destruição do Grafo).
- [ ] **Implementações Práticas Comentadas**:
  - Código demonstrando a anatomia de tensores, dispositivos e transferências assíncronas.
  - Demonstração prática do produto vetor-Jacobiano com tensores não-escalares passando o argumento `gradient` em `backward()`.
  - Demonstração do efeito de acúmulo de gradientes e zeramento explícito (`zero_grad(set_to_none=True)`).
  - Comparação de benchmarks de memória e performance entre `torch.no_grad()`, `torch.inference_mode()` e manipulação com `detach()`.
  - Exemplo prático de atualização manual de parâmetros e uso de hooks de gradiente (`register_hook`).
- [ ] **Armadilhas & Mitigações**: Mapeamento explícito de memory leaks clássicos em loops de treino (ex.: `total_loss += loss` vs `loss.item()`) e suas implicações críticas para o particionamento e simulação de Aprendizado Federado.
- [ ] **Cheatsheet Atualizado**: Snippets essenciais de manipulação de tensores e autograd documentados em `subjects/aprendizado-federado/cheatsheets/quick-reference.md`.
