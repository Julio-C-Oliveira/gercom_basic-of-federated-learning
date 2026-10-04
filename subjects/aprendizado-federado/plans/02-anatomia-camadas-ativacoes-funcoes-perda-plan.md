# Plano de Pesquisa: Anatomia de Camadas, Ativações e Funções de Perda

**Assunto**: `Aprendizado Federado` (`aprendizado-federado`)  
**Tópico ID**: `T02` | **Fase**: `01. Nivelamento PyTorch`  
**Domínio / Eixo Temático**: Deep Learning com PyTorch / Arquitetura Neural e Funções de Otimização  
**Data**: 2026-10-04  
**Status**: Planejado  

---

## 1. Objetivo da Investigação & Metas de Aprendizado

O objetivo deste tópico é aprofundar os blocos de construção fundamentais do PyTorch fornecidos pelo módulo `torch.nn`, capacitando o pesquisador a projetar, instanciar e diagnosticar arquiteturas neurais robustas, computacionalmente eficientes e adaptadas aos desafios singulares do **Aprendizado Federado (FL)**. Ao final desta investigação, o pesquisador deverá:

1. Compreender a anatomia interna de camadas lineares (`nn.Linear`) e convolucionais (`nn.Conv2d`), dominando o cálculo analítico de dimensões espaciais de saída, alocação de parâmetros e estratégias canônicas de inicialização (Kaiming/He vs. Xavier/Glorot).
2. Compreender a mecânica matemática das camadas de normalização (`BatchNorm2d`, `LayerNorm`, `GroupNorm`) e desvendar por que o `BatchNorm` sofre degradação severa em cenários federados não-IID com lotes locais pequenos, justificando a adoção de `GroupNorm` e `LayerNorm` em FL.
3. Avaliar as propriedades não-lineares, fluxo de gradientes e suscetibilidade à saturação das funções de ativação clássicas e modernas (ReLU, LeakyReLU, GELU, Sigmoid, Softmax).
4. Dominar os critérios de perda supervisionados (`nn.CrossEntropyLoss`, `nn.BCEWithLogitsLoss`, `nn.MSELoss`), compreendendo a estabilidade numérica contra *underflow/overflow* (LogSumExp trick) e o tratamento de classes desbalanceadas através de pesos ponderados.
5. Desenvolver a capacidade de calcular e inspecionar a pegada de parâmetros de uma rede neural (`model.parameters()`), quantificando o payload de comunicação em bytes necessário para cada rodada de agregação federada.

---

## 2. Perguntas-Chave de Investigação

As respostas a estas perguntas constituirão o núcleo do documento de pesquisa:

1. **Camadas Lineares e Convolucionais (`nn.Linear`, `nn.Conv2d`)**:
   - Como operam os tensores de pesos e vieses em `nn.Linear` ($Y = XW^T + b$) e `nn.Conv2d`?
   - Qual a fórmula analítica de dimensionalidade de saída para convoluções 2D em função de $H_{in}, W_{in}$, `kernel_size`, `stride`, `padding` e `dilation`?
   - Como o número de canais e parâmetros impacta diretamente o tráfego de rede durante o upload de modelos em clientes federados?
2. **Normalizações e a Crise do BatchNorm em Aprendizado Federado**:
   - Como operam matematicamente `BatchNorm2d`, `LayerNorm` e `GroupNorm` (sobre quais eixos média e variância são calculadas em cada uma)?
   - Por que o `BatchNorm` depende de estatísticas móveis (`running_mean`, `running_var`) e por que a média aritmética dessas estatísticas falha catastroficamente no agregador central do FedAvg sob dados não-IID?
   - Por que `GroupNorm` e `LayerNorm` eliminam a dependência de lote e preservam a equivalência de representação em clientes com distribuições heterogêneas?
3. **Propriedades e Dinâmica de Gradientes das Funções de Ativação**:
   - Quais as características matemáticas de ReLU, LeakyReLU, GELU, Sigmoid e Softmax?
   - O que é o fenômeno de *Dying ReLU* e como a ativação estocástica aproximada GELU (Gaussian Error Linear Unit) mitiga esse gargalo em arquiteturas modernas?
   - Por que funções com saturação (Sigmoid/Tanh) causam desaparecimento de gradiente nas camadas iniciais e em que circunstâncias específicas seu uso ainda é mandatório?
4. **Critérios de Perda e Estabilidade Numérica**:
   - Como o PyTorch otimiza internamente `nn.CrossEntropyLoss` combinando `nn.LogSoftmax` e `nn.NLLLoss` em um único kernel fundido (*LogSumExp trick*)?
   - Por que aplicar uma camada explícita `nn.Softmax` antes de `nn.CrossEntropyLoss` resulta em instabilidade numérica e degradação de gradientes?
   - Como parametrizar `weight` em `CrossEntropyLoss` e `pos_weight` em `BCEWithLogitsLoss` para compensar o desbalanceamento severo de classes local em clientes federados heterogêneos?
5. **Inicialização de Parâmetros e Sincronização Inicial em FL**:
   - Qual a diferença matemática entre inicialização Kaiming/He (adequada para ativações ReLU/GELU) e Xavier/Glorot (adequada para ativações simétricas/lineares)?
   - Por que no Aprendizado Federado é obrigatório que todos os clientes iniciem a Rodada 1 a partir da **exata mesma semente/inicialização global de pesos** gerada pelo servidor central?
6. **Diagnóstico e Métricas de Comunicação do Modelo**:
   - Como inspecionar programaticamente os parâmetros de cada camada (`shape`, `numel`, `element_size`)?
   - Como calcular o payload exato em Megabytes (MB) que será transmitido via gRPC/rede por cada cliente a cada rodada de treinamento federado?

---

## 3. Bibliotecas, Frameworks & Recursos Necessários

| Ferramenta / Biblioteca | Versão Mínima | Propósito no Estudo | Fonte / Instalação |
| :--- | :--- | :--- | :--- |
| `torch` | `>=2.1.0` | Instanciação de camadas (`torch.nn`), ativações e funções de perda | `pip install torch` |
| `torchvision` | `>=0.16.0` | Datasets canônicos e modelos de referência para experimentação | `pip install torchvision` |
| `torchinfo` | `>=1.8.0` | Geração de resumos detalhados da arquitetura, contagem de parâmetros e memória de ativação | `pip install torchinfo` |
| `matplotlib` | `>=3.8.0` | Visualização gráfica de curvas de ativação, gradientes e curvas de perda | `pip install matplotlib` |

---

## 4. Ambiente de Teste, Dataset & Cenário Prático

- **Ambiente Recomendado**: Script Python standalone ou Jupyter Notebook em ambiente Linux com suporte a PyTorch.
- **Cenário Prático 1 (Inspeção de Normalizações em FL)**:
  - Implementação de duas variantes de uma CNN (Convolução + Normalização + ReLU + Pooling):
    - Variante A: com `nn.BatchNorm2d`
    - Variante B: com `nn.GroupNorm(num_groups=..., num_channels=...)`
  - Simulação de 2 clientes heterogêneos com dados sintéticos ou subconjuntos desbalanceados de CIFAR-10 com batch size pequeno ($B = 4$ ou $8$).
  - Demonstração empírica do desvio de `running_mean` e `running_var` no BatchNorm vs. a estabilidade independente de batch do GroupNorm.
- **Cenário Prático 2 (Pipeline de Perda e Logits)**:
  - Construção de um classificador multi-classe com `nn.Sequential` demonstrando a saída de logits brutos sem ativação final.
  - Demonstração do uso de pesos de classe (`class_weights`) para tratar distribuições de rótulos não-IID (ex.: cliente A possui 90% classe 0 e 10% classe 1).
  - Cálculo analítico e programático do payload de upload do modelo em bytes.
- **Instruções rápidas de setup**:
  ```bash
  pip install torch torchvision torchinfo matplotlib
  ```

---

## 5. Critérios de Aceite para Conclusão

Para que a documentação de pesquisa deste tópico seja aprovada pelo comando `/verify`:

- [ ] **Rigor Teórico & Matemático**: Equações matemáticas formais de `nn.Linear`, `nn.Conv2d`, fórmulas de dimensionalidade de saída, e formulações de `BatchNorm`, `LayerNorm`, `GroupNorm`, ReLU, GELU e `CrossEntropyLoss`.
- [ ] **Análise Crítica de FL**: Discussão aprofundada demonstrando matematicamente e conceitualmente o colapso do `BatchNorm` sob dados não-IID e por que `GroupNorm` é a solução recomendada em papers canônicos de FL.
- [ ] **Implementações Práticas Comentadas**:
  - Código comparativo de normalizações (BatchNorm vs GroupNorm vs LayerNorm) em mini-batches pequenos.
  - Implementação demonstrando a estabilidade numérica do *LogSumExp trick* em `CrossEntropyLoss` vs perda calculada manualmente com `Softmax + log`.
  - Função utilitária para cálculo da pegada de parâmetros e payload de comunicação federada em MB.
- [ ] **Walkthrough Reproduzível**: Script autônomo comprovando a estabilidade de treinamento sob dados heterogêneos ao adotar `GroupNorm`.
- [ ] **Cheatsheet Atualizado**: Snippets essenciais de camadas, substituição de normalização para FL e cálculo de parâmetros inseridos em `subjects/aprendizado-federado/cheatsheets/quick-reference.md`.
