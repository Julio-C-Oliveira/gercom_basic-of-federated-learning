# Roadmap de Estudo: Aprendizado Federado

**Slug**: `aprendizado-federado`  
**Domínio / Eixo Temático**: Machine Learning Distribuído / Aprendizado Federado  
**Linha de Pesquisa / Subtema**: Deep Learning com PyTorch, Algoritmos de Agregação e Orquestração com Flower & NVFlare  
**Status Geral**: Em Andamento  
**Criado em**: 2026-10-04 | **Última Atualização**: 2026-10-04  
**Cheatsheet Rápido**: [quick-reference.md](cheatsheets/quick-reference.md)

---

## 1. Visão Geral & Escopo

O objetivo deste módulo é estabelecer uma base sólida, científica e prática sobre **Aprendizado Federado (Federated Learning - FL)**, capacitando o pesquisador a projetar, simular, implementar, customizar e comparar ecossistemas federados com rigor conceitual e metodológico.

O estudo é estruturado em etapas integradas:
1. **Nivelamento Prático em Deep Learning com PyTorch**: Domínio do mecanismo de grafos dinâmicos (`autograd`), anatomia e tipologia de camadas neurais, funções de ativação, funções de perda, otimizadores e o ciclo completo de treinamento/validação local.
2. **Fundamentos e Algoritmos de Aprendizado Federado**: Compreensão aprofundada da arquitetura cliente-servidor, do ciclo de rodadas federadas, da matemática do algoritmo canônico **FedAvg**, dos gargalos de comunicação e dos desafios críticos de **heterogeneidade estatística (dados não-IID)** com algoritmos mitigadores (FedProx, SCAFFOLD).
3. **Ecossistemas e Frameworks Práticos (Flower & NVFlare)**: Exploração aprofundada dos dois principais frameworks da atualidade:
   - **Flower (flwr)**: Flexibilidade, arquitetura moderna (`ClientApp`, `ServerApp`, `SuperLink`), simulação escalável com alocação virtual de recursos (Ray).
   - **NVIDIA NVFlare**: Arquitetura corporativa orientada a componentes (`Controller`/`Worker`, `FedJob`, `ModelLearner`), segurança e suporte de alta performance.
4. **Customização de Componentes e Extensibilidade**: Técnicas práticas para modificar e estender as funções nativas dos frameworks: implementação de funções de agregação personalizadas (pesos adaptativos, médias aparadas, métricas locais), políticas de seleção e amostragem customizada de clientes (por recursos, latência ou importância estatística) e interceptores/filtros nos dois frameworks.
5. **Análise Comparativa, Segurança & Métricas**: Trade-offs entre Flower e NVFlare, privacidade diferencial (DP), agregação segura e métricas de benchmarking federado.

- **Ambiente / Setup Recomendado**:
  - Python 3.10+
  - PyTorch 2.x (com suporte a CUDA/CPU)
  - Flower (`flwr>=1.8.0`, `flwr-datasets`)
  - NVIDIA NVFlare (`nvflare>=2.4.0`)
  - Ray (para simulações de múltiplos nós em máquina única)
  - JupyterLab / VS Code / Linux CLI
- **Pré-requisitos**:
  - Programação intermediária/avançada em Python e manipulação de tensores (NumPy)
  - Fundamentos de cálculo diferencial (derivadas e gradientes) e álgebra linear

---

## 2. Tabela de Controle de Tópicos

| ID | Fase / Eixo Temático | Tópico | Status | Plano Investigativo | Documentação de Pesquisa |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **T01** | 01. Nivelamento PyTorch | **Fundamentos de Redes Neurais e Autograd no PyTorch**<br>Tensores, dispositivos (CPU/CUDA), grafos computacionais dinâmicos, forward pass, cálculo de gradientes e retropropagação (`loss.backward()`). | [x] | [Plano](plans/01-fundamentos-redes-neurais-autograd-pytorch-plan.md) | [Pesquisa](research/01-fundamentos-redes-neurais-autograd-pytorch.md) |
| **T02** | 01. Nivelamento PyTorch | **Anatomia de Camadas, Ativações e Funções de Perda**<br>Camadas Lineares (`nn.Linear`), Convolucionais (`nn.Conv2d`), Normalizações (`BatchNorm`, `LayerNorm`, `GroupNorm` e impacto em FL); Funções de ativação (ReLU, GELU, Sigmoid, Softmax); Critérios de perda (`CrossEntropyLoss`, `MSELoss`). | [x] | [Plano](plans/02-anatomia-camadas-ativacoes-funcoes-perda-plan.md) | [Pesquisa](research/02-anatomia-camadas-ativacoes-funcoes-perda.md) |
| **T03** | 01. Nivelamento PyTorch | **Pipeline Completo de Treinamento e Otimização com PyTorch**<br>Estruturação com `nn.Module`, manipulação de dados via `Dataset` e `DataLoader`, otimizadores (`SGD`, `Adam`), decaimento de aprendizado e o ciclo canônico de treinamento/avaliação. | [x] | [Plano](plans/03-pipeline-treinamento-otimizacao-pytorch-plan.md) | [Pesquisa](research/03-pipeline-treinamento-otimizacao-pytorch.md) |
| **T04** | 02. Fundamentos de FL | **O Paradigma do Aprendizado Federado e Arquitetura Distribuída**<br>Motivação, privacidade e soberania de dados; Topologia Servidor-Clientes vs Descentralizada; Ciclo de rodadas federadas (Broadcast, Local Training, Upload de Pesos/Gradientes, Agregação). | [ ] | - | - |
| **T05** | 02. Fundamentos de FL | **O Algoritmo Canônico Federated Averaging (FedAvg)**<br>Formulação matemática e derivação do FedAvg (McMahan et al.); Média ponderada por tamanho de amostra local ($n_k$); Impacto das épocas locais ($E$), batch size ($B$) e taxa de amostragem de clientes ($C$). | [ ] | - | - |
| **T06** | 02. Fundamentos de FL | **Heterogeneidade Estatística (Dados Não-IID) e Mitigações**<br>Tipos de não-IID (skew de classes, skew de atributos, concept drift); O problema do *Client Drift*; Particionamento sintético via Distribuição Dirichlet ($\alpha$); Algoritmos avançados de mitigação (FedProx com termo proximal e SCAFFOLD com variáveis de controle). | [ ] | - | - |
| **T07** | 03. Framework Flower | **Arquitetura, Core e APIs do Framework Flower (flwr)**<br>Filosofia agnóstica do Flower; Arquitetura moderna (`ClientApp`, `ServerApp`, `SuperLink`, `SuperNode`); Criação de clientes via `NumPyClient`/`Client`; Estratégias nativas e customizadas herdadas de `flwr.server.strategy.Strategy`. | [ ] | - | - |
| **T08** | 03. Framework Flower | **Simulação em Larga Escala e Alocação de Recursos com Flower**<br>Uso da engine de simulação (`flwr run` / `flwr.simulation`); Gerenciamento virtual de recursos CPU/GPU com Ray; Particionamento federado padronizado com `flwr-datasets`. | [ ] | - | - |
| **T09** | 04. Framework NVFlare | **Arquitetura Corporativa e Componentes do NVIDIA NVFlare**<br>Conceitos fundamentais: `Controller`, `Worker`, `ModelLearner`, `FLComponent`; Filosofia de orquestração orientada a eventos; Modos de execução (POC, FL Simulator e Produção). | [ ] | - | - |
| **T10** | 04. Framework NVFlare | **Workflows, APIs e Execução de FedJobs no NVFlare**<br>Estruturação de um `FedJob`; Implementação do cliente com Client-Side API (`ClientAPI` vs `Executor`); Workflows de servidor (`ScatterAndGather`, `CrossSiteEval`); Filtros de dados e serialização de parâmetros. | [ ] | - | - |
| **T11** | 05. Customização & Extensibilidade | **Customização Avançada: Agregação, Seleção de Clientes e Hooks Customizados**<br>Como modificar e estender componentes nativos: Funções de agregação personalizadas (agregação ponderada por acurácia/perda, defesas bizantinas robustas como Trimmed Mean e Coordinate-wise Median); Políticas de seleção e amostragem de clientes (amostragem por importância, restrições de latência/bateria, tratamento de *stragglers*); Implementação prática no Flower (sobrescrevendo `configure_fit`, `aggregate_fit`, `fit_metrics_aggregation_fn`) e no NVFlare (custom `Aggregator`, custom `Controller`/`Selector` e `Filters`). | [ ] | - | - |
| **T12** | 06. Comparativo & Análise | **Estudo Comparativo Prático: Flower vs NVFlare**<br>Análise comparativa multidimensional: facilidade de prototipação, escalabilidade, acoplamento de hardware (GPUs NVIDIA), prontidão enterprise/médica, depuração e curvas de aprendizado. | [ ] | - | - |
| **T13** | 06. Segurança & Métricas | **Privacidade, Segurança e Benchmarking em Sistemas Federados**<br>Ameaças e ataques (Data/Model Poisoning, Inversão de Gradientes); Mecanismos de defesa: Privacidade Diferencial (DP - Local e Global) e Agregação Segura (SecAgg); Métricas de convergência, overhead de rede e equidade (*fairness*). | [ ] | - | - |

---

## 3. Métricas de Progresso

- **Total de Tópicos**: 13
- **Concluídos**: 3
- **Pendentes**: 10
- **Progresso**: 23.1%

---

## 4. Ambientes Práticos, Datasets & Recursos Recomendados

- [x] **Documentação PyTorch**: [pytorch.org/docs](https://pytorch.org/docs/stable/index.html) - Tutoriais de autograd, `torch.nn` e rotinas de treino.
- [ ] **Documentação Oficial do Flower**: [flower.ai/docs](https://flower.ai/docs/) - Guias da arquitetura Flower Next e customização de estratégias (`Strategy`).
- [ ] **Documentação Oficial do NVIDIA NVFlare**: [nvflare.readthedocs.io](https://nvflare.readthedocs.io/) - Guia de programação de controladores, agregadores customizados e filtros.
- [ ] **Paper Base FedAvg**: McMahan et al., *"Communication-Efficient Learning of Deep Networks from Decentralized Data"*, AISTATS 2017.
- [ ] **Paper FedProx**: Li et al., *"Federated Optimization in Heterogeneous Networks"*, MLSys 2020.
- [ ] **Paper Robust Aggregation**: Blanchard et al., *"Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"*, NeurIPS 2017.
- [ ] **Datasets de Benchmark Recomendados**:
  - `MNIST` e `Fashion-MNIST`: Validação rápida e nivelamento de pipeline.
  - `CIFAR-10`: Experimentação de dados não-IID com partições Dirichlet ($\alpha \in \{0.1, 0.5, 1.0\}$).
  - `Femnist` ou `Shakespeare` (LEAF benchmark): Teste de heterogeneidade realística.

---

## 5. Histórico de Sessões

- **2026-10-04**: Inicialização do subject `aprendizado-federado` via `/define-subject`.
- **2026-10-04**: Inclusão do tópico dedicado **T11 (Customização Avançada: Agregação, Seleção de Clientes e Hooks Customizados)** para cobrir extensibilidade e modificação de funções nativas no Flower e NVFlare, totalizando 13 tópicos no roadmap.
- **2026-10-04**: Elaboração do plano investigativo detalhado para o tópico **T01 (Fundamentos de Redes Neurais e Autograd no PyTorch)** via `/define-research-topic`.
- **2026-10-04**: Tópico **T01 (Fundamentos de Redes Neurais e Autograd no PyTorch)** concluído via `/do-research`.
- **2026-10-04**: Elaboração do plano investigativo detalhado para o tópico **T02 (Anatomia de Camadas, Ativações e Funções de Perda)** via `/define-research-topic`.
- **2026-10-04**: Tópico **T02 (Anatomia de Camadas, Ativações e Funções de Perda)** concluído via `/do-research`.
- **2026-10-04**: Elaboração do plano investigativo detalhado para o tópico **T03 (Pipeline Completo de Treinamento e Otimização com PyTorch)** via `/define-research-topic`.
- **2026-10-04**: Tópico **T03 (Pipeline Completo de Treinamento e Otimização com PyTorch)** concluído via `/do-research`.
