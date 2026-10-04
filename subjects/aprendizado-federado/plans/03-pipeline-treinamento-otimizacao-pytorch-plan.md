# Plano de Pesquisa: Pipeline Completo de Treinamento e Otimização com PyTorch

**Assunto**: `Aprendizado Federado` (`aprendizado-federado`)  
**Tópico ID**: `T03` | **Fase**: `01. Nivelamento PyTorch`  
**Domínio / Eixo Temático**: Deep Learning com PyTorch / Engenharia de Treinamento, Dados e Otimização  
**Data**: 2026-10-04  
**Status**: Planejado  

---

## 1. Objetivo da Investigação & Metas de Aprendizado

O objetivo deste tópico é dominar a engenharia do ciclo de vida completo de treinamento e avaliação no PyTorch, articulando as abstrações orientadas a objetos (`nn.Module`), ingestão e particionamento de dados (`Dataset`, `DataLoader`) e motores de otimização estocástica (`torch.optim`). Ao final desta investigação, o pesquisador deverá:

1. Estruturar modelos complexos utilizando as melhores práticas de `nn.Module`, dominando a hierarquia de parâmetros treináveis (`nn.Parameter`), buffers não-treináveis (`register_buffer`), e manipulação avançada de dicionários de estado (`state_dict` e `load_state_dict`).
2. Projetar pipelines de dados escaláveis e reprodutíveis via `Dataset` e `DataLoader`, compreendendo o uso de `torch.utils.data.Subset` para particionar coleções globais em fatias isoladas de clientes federados, além de otimizar parâmetros de throughput (`num_workers`, `pin_memory`, `persistent_workers`).
3. Dominar a formulação matemática e comportamento empírico dos otimizadores estocásticos canônicos: **SGD** (com Momentum clássico de Polyak e Nesterov) e **AdamW** (com momento de 2ª ordem e decaimento de peso desacoplado).
4. Compreender as armadilhas do estado interno de otimizadores em Aprendizado Federado: o dilema entre reiniciar os buffers de momento a cada rodada local vs. acumular momentum no servidor central (**Server Momentum / FedAvgM**).
5. Integrar agendadores de taxa de aprendizado (`torch.optim.lr_scheduler`), mapeando a dinâmica de decaimento por época local vs. rodada global federada.
6. Construir módulos reutilizáveis e autocontidos de treino (`train_one_epoch`) e validação (`evaluate`), projetados com desacoplamento rigoroso para servir como blocos fundamentais de nós clientes em frameworks federados (Flower e NVFlare).

---

## 2. Perguntas-Chave de Investigação

As respostas a estas perguntas constituirão o núcleo do documento de pesquisa:

1. **Arquitetura `nn.Module` e Gestão de Estados**:
   - Qual a diferença exata entre um parâmetro treinável (`nn.Parameter`), um atributo comum de instância e um buffer registrado (`self.register_buffer(...)`)?
   - Como o `state_dict()` empacota parâmetros e buffers, e quais os cuidados ao aplicar `load_state_dict(..., strict=False)` durante a substituição de pesos globais em clientes federados?
2. **Engenharia de Dados com `Dataset` e `DataLoader`**:
   - Como criar uma classe `Dataset` customizada e como fatiar datasets canônicos (ex.: CIFAR-10/Fashion-MNIST) em partições não-IID isoladas utilizando `torch.utils.data.Subset`?
   - Qual o papel de `collate_fn`, `num_workers`, `pin_memory` e `drop_last` no rendimento computacional e na estabilidade de batches pequenos em clientes com recursos restritos?
3. **Mecânica Matemática dos Otimizadores (SGD vs. AdamW)**:
   - Quais as equações completas de atualização de parâmetros para SGD com momentum e aceleração de Nesterov?
   - Por que o `AdamW` (Loshchilov & Hutter) desacopla o decaimento de peso (*weight decay*) do gradiente com momento de 2ª ordem, corrigindo a penalidade L2 original do Adam?
4. **O Dilema do Momentum Local vs. Server Momentum em FL**:
   - Em clientes federados que executam $E$ épocas locais, o que acontece com os tensores de momento do otimizador entre rodadas consecutivas?
   - Se o cliente mantiver seu buffer de momento local entre rodadas, como isso induz ao *Client Drift* sob dados heterogêneos?
   - Como o algoritmo **FedAvgM (Federated Averaging with Server Momentum)** soluciona esse problema centralizando o momento no agregador global?
5. **Agendamento de Taxa de Aprendizado (Learning Rate Schedulers)**:
   - Como orquestrar agendadores como `CosineAnnealingLR` ou `StepLR` em sistemas federados: o decaimento deve ser regido pelas rodadas globais do servidor ou pelas épocas locais de cada cliente?
6. **Padronização do Ciclo de Treino e Validação**:
   - Como escrever funções puras de treino local (`train_one_epoch`) e teste (`evaluate`) que retornem dicionários estruturados de métricas (loss, acurácia top-1, throughput de amostras/s) com garantia estrita de desalocação de tensores temporários?

---

## 3. Bibliotecas, Frameworks & Recursos Necessários

| Ferramenta / Biblioteca | Versão Mínima | Propósito no Estudo | Fonte / Instalação |
| :--- | :--- | :--- | :--- |
| `torch` | `>=2.1.0` | `nn.Module`, `torch.optim`, `torch.utils.data` e schedulers | `pip install torch` |
| `torchvision` | `>=0.16.0` | Datasets padronizados (`Fashion-MNIST` / `CIFAR-10`) e transformações | `pip install torchvision` |
| `numpy` | `>=1.24.0` | Manipulação de índices de partição e cálculo analítico de métricas | `pip install numpy` |
| `matplotlib` | `>=3.8.0` | Visualização de curvas comparativas de convergência (SGD vs AdamW) | `pip install matplotlib` |

---

## 4. Ambiente de Teste, Dataset & Cenário Prático

- **Ambiente Recomendado**: Script Python standalone em terminal Linux ou notebook interativo.
- **Cenário Prático 1 (Particionamento e Dataloaders Federados)**:
  - Download do dataset `Fashion-MNIST`.
  - Particionamento sintético dos índices em 3 fatias de clientes heterogêneas utilizando `torch.utils.data.Subset`.
  - Configuração de `DataLoader` para cada partição simulando 3 clientes locais independentes.
- **Cenário Prático 2 (Loop Completo de Treinamento e Avaliação)**:
  - Implementação de um classificador convolucional modular com `nn.Module`.
  - Execução de rotinas desacopladas `train_one_epoch()` e `evaluate()`.
  - Teste comparativo entre SGD (com momentum local reiniciado a cada rodada) vs. AdamW.
  - Implementação de um agendador de taxa de aprendizado federado por rodada global.
- **Instruções rápidas de setup**:
  ```bash
  pip install torch torchvision numpy matplotlib
  ```

---

## 5. Critérios de Aceite para Conclusão

Para que a documentação de pesquisa deste tópico seja aprovada pelo comando `/verify`:

- [ ] **Rigor Teórico & Matemático**: Equações matemáticas completas do SGD (momentum e Nesterov) e AdamW, com explicação formal do desacoplamento de weight decay.
- [ ] **Análise Crítica de FL**: Discussão aprofundada sobre a persistência de estados de otimizadores em clientes federados, justificando por que o *Server Momentum* (FedAvgM) é a arquitetura canônica recomendada.
- [ ] **Pipeline de Dados Reproduzível**: Implementação prática de particionamento de datasets via `Subset` e `DataLoader` simulando dados heterogêneos de múltiplos clientes.
- [ ] **Funções Modulares de Treino e Validação**: Código de `train_one_epoch()` e `evaluate()` retornando métricas estruturadas, com uso de `set_to_none=True`, `inference_mode()` e cálculo de acurácia top-1.
- [ ] **Walkthrough Completo**: Script funcional demonstrando a simulação de 2 rodadas de treino local em um cliente fictício, persistência de `state_dict` e avaliação global.
- [ ] **Cheatsheet Atualizado**: Snippets essenciais de loop canônico, particionamento com `Subset/DataLoader`, schedulers e exportação/carregamento de `state_dict` adicionados a `quick-reference.md`.
