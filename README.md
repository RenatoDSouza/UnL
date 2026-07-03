# QfederatedUlearning

Projeto base para Quantum Federated Learning (QFL) com dois experimentos separados:

1. Treinamento puro de modelo global em uma federação com um servidor central e cinco clientes.
2. Machine unlearning com desconexão de um cliente e remoção da sua influência histórica via Quantum Fisher Information (QFI).

## Organização

```text
.
├── experiments
│   ├── qfl_training
│   │   ├── configs
│   │   ├── logs
│   │   ├── outputs
│   │   └── scripts
│   └── qfl_unlearning_qfi
│       ├── configs
│       ├── logs
│       ├── outputs
│       └── scripts
├── src
│   └── qfl
│       ├── common
│       ├── data
│       ├── federated
│       ├── quantum
│       └── utils
└── README.md
```

## Objetivo arquitetural

- `src/qfl/common`: contratos, tipos, validações e constantes compartilhadas.
- `src/qfl/data`: ingestão, particionamento e pré-processamento do FEMNIST.
- `src/qfl/federated`: abstrações do servidor central, clientes e estratégia de agregação.
- `src/qfl/quantum`: circuito quântico, encoder, métrica e rotinas relacionadas a QFI.
- `src/qfl/utils`: logging, seed, serialização e helpers de engenharia.
- `experiments/qfl_training`: fluxo do experimento de QFL puro.
- `experiments/qfl_unlearning_qfi`: fluxo do experimento de unlearning baseado em QFI.

## Stack

- `PennyLane` como biblioteca quântica principal.
- `lightning.gpu` como backend preferencial, com fallback automático para `lightning.qubit` e `default.qubit`.
- `PyTorch` e `NumPy` para suporte numérico e integração futura com pipelines híbridos.

## Dados

O repositório assume um arquivo local `data/femnist_sample.npz` com chaves `x` e `y` para facilitar execução controlada do pipeline.
A ingestão completa do FEMNIST do LEAF deve ser adicionada em seguida como etapa dedicada de download e parsing do dataset.

## Próximos passos

1. Adicionar o downloader/parsing do FEMNIST do LEAF no formato original.
2. Refinar o modelo quântico com codificação mais fiel ao caso de uso e treino por gradiente.
3. Criar testes automatizados para agregação, particionamento e rotina de unlearning.

## Execução

Os scripts de referência estão em:

- `experiments/qfl_training/scripts/run_training.py`
- `experiments/qfl_unlearning_qfi/scripts/run_unlearning.py`

Ambos assumem um arquivo local `data/femnist_sample.npz` durante esta fase inicial do projeto.
