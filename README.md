# QfederatedUlearning

Implementação base de **Quantum Federated Learning (QFL)** com foco em dois experimentos:

1. **Treinamento federado quântico puro**
   - Um servidor central coordena cinco clientes.
   - Os clientes treinam localmente um modelo quântico em `PennyLane`.
   - O servidor agrega as atualizações para produzir o modelo global.

2. **Machine unlearning com SHAP + QFI**
   - O mesmo ambiente federado é treinado inicialmente.
   - Um cliente é desconectado da federação.
   - A influência histórica desse cliente é reduzida por um fluxo híbrido com SHAP e Matriz de Fisher Quântica.

O repositório foi reorganizado para separar responsabilidades:

- `src/` contém a lógica reutilizável e o pipeline central.
- `experiments/` contém apenas wrappers de execução, configs, logs e outputs.
- `tests/` contém validação automatizada do comportamento do núcleo.

Um levantamento detalhado das mudanças locais mais recentes em relação à
`origin/main`, incluindo decisões sobre datasets e publicação, está em
[`docs/CHANGES_FROM_E97D0CE.md`](docs/CHANGES_FROM_E97D0CE.md).

## Estrutura do projeto

```text
.
├── experiments/
│   ├── qfl_training/
│   │   ├── configs/
│   │   ├── logs/
│   │   ├── outputs/
│   │   └── scripts/
│   └── qfl_unlearning_qfi/
│       ├── configs/
│       ├── logs/
│       ├── outputs/
│       └── scripts/
├── src/
│   └── qfl/
│       ├── common/
│       ├── data/
│       ├── experiments/
│       ├── federated/
│       ├── quantum/
│       └── utils/
├── tests/
├── pyproject.toml
└── README.md
```

## Stack técnica

- **Linguagem**: Python 3.10+
- **Biblioteca quântica**: `PennyLane`
- **Backend quântico preferencial**: `lightning.gpu`
- **Fallbacks automáticos**:
  - `lightning.qubit`
  - `default.qubit`
- **Bibliotecas auxiliares**:
  - `NumPy`
  - `PyTorch`
  - `pandas`
  - `scikit-learn`

## O que este projeto faz hoje

Nesta fase, o repositório contém:

- arquitetura de servidor e cliente federado;
- modelo quântico em `PennyLane`;
- rotina de agregação do servidor;
- rotina de unlearning híbrida baseada em SHAP + QFI;
- orquestrador central em `src/qfl/experiments/pipeline.py`;
- wrappers finos de execução para treino e unlearning;
- suporte a múltiplas seeds com agregação estatística;
- estatísticas agregadas com média, desvio-padrão, mediana, min/max e CI aproximado;
- baselines comparativos de unlearning: `no_unlearning`, `shap_only`, `qfi_only`, `shap_qfi`;
- exportação de logs em JSON e CSV por seed;
- ablation study sobre `angle`, `iqp` e `reupload`;
- testes mínimos para validar particionamento, agregação e unlearning.

## Requisitos de ambiente

### 1. Python

Use Python 3.10 ou superior.

Verifique com:

```bash
python3 --version
```

### 2. GPU

O projeto tenta usar GPU via `lightning.gpu` quando disponível.

Importante:

- ter drivers NVIDIA compatíveis;
- ter CUDA corretamente instalado;
- ter uma versão de `PennyLane` e `pennylane-lightning` compatível com GPU;
- se a GPU não estiver disponível, o projeto faz fallback automático para CPU.

### 3. Ambiente virtual

É recomendável usar `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

No Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

## Instalação

Instale as dependências a partir do `requirements.txt`:

```bash
pip install -r requirements.txt
```

Se preferir instalar o pacote local em modo editável:

```bash
pip install -e .
```

## Dados

O projeto utiliza `flwr-datasets[vision]` para baixar e gerenciar o conjunto FEMNIST (`flwrlabs/femnist`).

- **Download automático**: acontece no primeiro `run`.
- **Particionamento**: usa `writer_id`, gerando partições não-IID.
- **Visualização**: há um script para inspeção de amostras em `data/visualize_femnist.py`.

## Como o código está organizado

### `src/`

Contém a lógica reutilizável:

- `qfl.data`: carregamento e particionamento do FEMNIST;
- `qfl.quantum`: modelo quântico e QFI;
- `qfl.federated`: cliente, servidor, estratégia, métricas e unlearning;
- `qfl.experiments`: pipeline único de treino e unlearning;
- `qfl.utils`: seed, checkpoint, I/O e progresso.

### `experiments/`

Contém wrappers de execução:

- lêem YAML;
- constroem dataclasses de configuração;
- chamam funções de `src/qfl/experiments/pipeline.py`;
- escrevem outputs e checkpoints.

### `tests/`

Contém testes automatizados para:

- particionamento;
- agregação;
- pipeline híbrido de unlearning;
- compatibilidade com o wrapper legado.

## Como rodar os experimentos

Os wrappers em `experiments/` são finos e chamam o pipeline central em `src/qfl/experiments/pipeline.py`.

### 1. Experimento de treinamento QFL

Script:

```text
experiments/qfl_training/scripts/run_training.py
```

Execute com:

```bash
python3 experiments/qfl_training/scripts/run_training.py
```

Fluxo:

1. Lê `experiments/qfl_training/configs/default.yaml`.
2. Monta `TrainingExperimentConfig`.
3. Chama `run_training_experiment()`.
4. Executa o experimento para cada seed em `seeds` ou, se ausente, usa `seed`.
5. Salva um resumo agregado em `experiments/qfl_training/outputs/training_summary.json`.
6. Exporta logs por seed em JSON e CSV para facilitar análise posterior.

### 2. Experimento de machine unlearning com SHAP + QFI

Script:

```text
experiments/qfl_unlearning_qfi/scripts/run_unlearning.py
```

Execute com:

```bash
python3 experiments/qfl_unlearning_qfi/scripts/run_unlearning.py
```

Fluxo:

1. Lê `experiments/qfl_unlearning_qfi/configs/default.yaml`.
2. Monta `UnlearningExperimentConfig`.
3. Chama `run_unlearning_experiment()`.
4. Executa o fluxo para cada seed em `seeds` ou, se ausente, usa `seed`.
5. Executa treino federado inicial.
6. Remove o cliente configurado em `excluded_client_id`.
7. Aplica unlearning híbrido SHAP + QFI.
8. Salva um resumo agregado em `experiments/qfl_unlearning_qfi/outputs/unlearning_summary.json`.
9. Exporta logs por seed em JSON e CSV.

## Configurações

Os valores padrão ficam nestes arquivos:

- `experiments/qfl_training/configs/default.yaml`
- `experiments/qfl_unlearning_qfi/configs/default.yaml`

Eles documentam parâmetros como:

- caminho do dataset;
- número de clientes;
- número de rodadas;
- cliente excluído no unlearning;
- lista de seeds para repetição estatística;
- lista de encodings para ablation study;
- número de data reuploads;
- preferência por GPU.

Esses YAMLs alimentam apenas a camada de execução. A lógica de negócio mora em `src/qfl/experiments/pipeline.py`.

## Encodings usados no estudo

| Encoding | Papel | Observação crítica |
| --- | --- | --- |
| `angle` | Baseline mais simples | Mais estável e barato, mas pode ter menor capacidade expressiva |
| `iqp` | Encoding com interações quânticas | Mais sensível ao número de wires e à escala de entrada |
| `reupload` | Data re-uploading | Aumenta a expressividade, mas também o custo e a chance de overfitting |

O `IQPEmbedding` merece atenção extra: se a dimensionalidade efetiva dos dados crescer ou a normalização não for consistente, o circuito pode ficar mais ruidoso e menos interpretável. Por isso ele deve ser analisado separadamente no ablation, e não apenas como uma média global.

O baseline `retrain_complete` é computacionalmente mais caro, mas é o comparador certo: ele treina do zero apenas com os clientes remanescentes e serve como referência para medir o custo/benefício real do unlearning.

## Saída dos experimentos

### Treinamento

Saída esperada:

```json
{
  "num_runs": 3,
  "rounds": [
    { "num_clients": 5.0 },
    { "num_clients": 5.0 },
    { "num_clients": 5.0 }
  ],
  "num_clients_mean": 5.0,
  "num_clients_std": 0.0
}
```

### Unlearning

Saída esperada:

```json
{
  "num_runs": 3,
  "runs": [],
  "baselines": {},
  "qfi_trace_before_mean": 0.0,
  "qfi_trace_before_std": 0.0,
  "qfi_trace_after_mean": 0.0,
  "qfi_trace_after_std": 0.0
}
```

O bloco `baselines` preserva os resultados comparativos por seed para as variantes `no_unlearning`, `shap_only`, `qfi_only` e `shap_qfi`. Isso é relevante porque um before/after isolado não sustenta uma conclusão causal forte sobre esquecimento.

O bloco `ablation` separa as estatísticas por encoding, permitindo comparar `angle`, `iqp` e `reupload` sem misturar arquiteturas distintas.

O baseline `retrain_complete` é o comparador mais importante para a avaliação do unlearning, porque representa o cenário ideal em que o cliente removido nunca participou do treino.

### Métricas de avaliação do unlearning

O experimento `qfl_unlearning_qfi` calcula métricas antes/depois para auditar o esquecimento:

#### 1. `forget_set_accuracy`

Mede a acurácia do modelo global sobre os dados do cliente removido da federação.

Interpretação desejada:

- quanto menor, melhor;
- idealmente deve se aproximar de uma taxa aleatória;
- indica redução efetiva da influência histórica do cliente.

#### 2. `retain_set_accuracy`

Mede a acurácia do modelo sobre os dados dos clientes que permanecem na federação.

Interpretação desejada:

- quanto maior, melhor;
- deve permanecer estável após o unlearning;
- evita perda catastrófica do conhecimento compartilhado.

#### 3. `mia_auc`

Mede a ROC-AUC de um membership inference attack por limiar de loss. Compara
amostras usadas no treino do cliente removido com amostras reservadas do mesmo
cliente que nunca participaram do treino.

Interpretação desejada:

- valores próximos de `0.5` indicam pouca evidência de pertencimento;
- valores próximos de `1.0` indicam maior vazamento ou memorização;
- após o unlearning, o objetivo é aproximar a métrica de `0.5` sem prejudicar o
  conjunto retido.

#### 4. `shap_drop_mean`

Mede a queda média de atribuição SHAP nos parâmetros/blocos alvo.

Interpretação desejada:

- quanto maior a queda nos blocos esquecidos, melhor;
- funciona como evidência de apagamento localizado.

## Estrutura e responsabilidades

- `src/`: núcleo reutilizável do projeto.
- `src/qfl/experiments/pipeline.py`: ponto único de orquestração para treino e unlearning.
- `experiments/`: scripts de execução, configs e artefatos.
- `experiments/*/outputs/logs/*.csv`: visão tabular dos runs por seed para análise externa.
- `tests/`: validação automatizada.

## Como deixar mais enxuto ainda

Se quiser reduzir redundância mais um passo, os cortes naturais são:

1. migrar também os wrappers de `experiments/` para um único entrypoint genérico;
2. extrair a montagem de dataset/cliente/servidor para uma única função pública;
3. remover o wrapper legado de unlearning quando não houver dependência externa dele;
4. adicionar `experiments/**/*.pyc` e saídas geradas ao `.gitignore` se ainda não estiverem cobertos.
# UnL
# UnL
