# Mudanças locais em relação à `origin/main`

Este documento descreve o trabalho local comparado ao commit `e97d0ce`
(`require PennyLane GPU device`), que era o `HEAD` de `origin/main` na última
verificação, em 14 de julho de 2026.

## Resumo do escopo

- 14 arquivos rastreados foram modificados no código, nas configurações e nos
  testes originais.
- O delta original contém 862 adições e 277 remoções.
- Dois caches FEMNIST estão presentes localmente, mas ainda não são rastreados:
  `data/femnist_bin01.npz` (436 KiB) e `data/femnist_dig012.npz` (576 KiB).
- O script `scripts/publish_branch.sh` e este documento foram acrescentados
  depois desse levantamento para organizar e publicar o conjunto de mudanças.

## Principais mudanças funcionais

### Dados FEMNIST e representação

- Corrige a binarização do FEMNIST por escritores. A regra anterior
  (`label > 0`) concentrava quase todas as amostras em uma única classe; a nova
  regra separa letras minúsculas das demais classes.
- Separa cada cliente em treino e avaliação. O conjunto de avaliação nunca é
  usado no treino e passa a representar os não membros na avaliação de
  membership inference.
- Adiciona PCA compartilhado e padronizado como alternativa às quatro médias de
  quadrantes.
- Adiciona três cenários experimentais:
  - `binary`: duas classes balanceadas e partição IID;
  - `noniid`: o cliente a esquecer possui uma classe positiva exclusiva;
  - `backdoor`: o cliente a esquecer possui uma associação de gatilho/canário
    exclusiva.
- Permite limitar o número de amostras por cliente para controlar o custo do
  cálculo da QFI.

### Treinamento federado

- Torna configuráveis a quantidade de camadas, as épocas locais, a taxa de
  aprendizado, a representação e a quantidade de atributos.
- O cliente passa a executar otimização local real sobre os pesos recebidos e a
  devolver `train_loss` e `train_accuracy`.
- O servidor agrega métricas locais ponderadas pelo número de amostras.
- Cada rodada passa a registrar `global_accuracy` e `global_loss` sobre uma
  amostra limitada do conjunto agregado dos clientes.
- Os pesos são normalizados para o formato esperado antes da agregação, evitando
  incompatibilidades entre vetores achatados e tensores do modelo.

### Unlearning SHAP + QFI

- O modelo de referência inicial agora é treinado com todos os clientes. O
  cliente excluído só é removido no baseline de retreinamento, evitando medir
  unlearning sobre um modelo que nunca viu os dados a esquecer.
- As atribuições SHAP são calculadas por grupos de parâmetros associados a cada
  fio/qubit, por permutações Monte Carlo.
- O esquecimento usa ascent no loss do conjunto a esquecer, opcionalmente
  mascarado por SHAP e precondicionado pela inversa da QFI.
- O ascent para antecipadamente quando a acurácia chega ao nível de um preditor
  desinformado, reduzindo o risco de degradar o modelo além do necessário.
- Passam a ser comparadas cinco referências: `no_unlearning`, `shap_only`,
  `qfi_only`, `shap_qfi` e `retrain_complete`.
- Os resultados são organizados por seed e encoding e agregados com média e
  desvio-padrão para uso nas tabelas do trabalho.

### Métricas e membership inference

- Substitui a taxa binária/proxy de sucesso da MIA por ROC-AUC de um ataque por
  limiar de loss por amostra.
- Membros são amostras de treino do cliente esquecido; não membros vêm do seu
  conjunto reservado de avaliação.
- A interpretação passa a ser: AUC próxima de 0,5 indica pouca evidência de
  pertencimento; valores próximos de 1,0 indicam maior vazamento/memorização.
- Acrescenta loss e acurácia antes/depois nos conjuntos de esquecimento e
  retenção, traço da QFI, passos de unlearning e densidade/efeito da máscara.

### Modelo quântico e dispositivos

- Garante que os pesos usados por `qml.metric_tensor` sejam diferenciáveis,
  evitando uma QFI silenciosamente degenerada após reconstruir o modelo com
  pesos agregados.
- Registra uma advertência quando o cálculo do tensor métrico falha.
- Quando GPU é solicitada, tenta `lightning.gpu` e faz fallback para
  `lightning.qubit` e `default.qubit`, em vez de encerrar imediatamente.
- As configurações padrão usam CPU (`prefer_gpu: false`), cinco rodadas, PCA com
  oito atributos, duas camadas e hiperparâmetros explícitos de treino e
  unlearning.

### Testes

- Atualiza o teste do relatório híbrido para a métrica MIA AUC e para o baseline
  desinformado.
- Valida o agrupamento dos parâmetros SHAP por fio/qubit.
- Valida que o fluxo de referência treina o modelo completo antes de aplicar os
  baselines de unlearning.

## Arquivos por área

| Área | Arquivos principais |
| --- | --- |
| Configuração | `experiments/qfl_training/configs/default.yaml`, `experiments/qfl_unlearning_qfi/configs/default.yaml` |
| Dados e pipeline | `src/qfl/data/femnist.py`, `src/qfl/experiments/pipeline.py` |
| Federação | `src/qfl/federated/client.py`, `server.py`, `strategy.py` |
| Unlearning e métricas | `src/qfl/federated/hybrid_unlearning.py`, `unlearning.py`, `mia.py`, `metrics.py` |
| Modelo quântico | `src/qfl/quantum/device.py`, `src/qfl/quantum/model.py` |
| Testes | `tests/test_hybrid_unlearning.py` |

## Decisão sobre os datasets locais

Os arquivos `.npz` somam aproximadamente 1 MiB e parecem ser caches
reproduzíveis pelos loaders. Por isso, `data/*.npz` foi incluído no `.gitignore`
e `scripts/publish_branch.sh` não os inclui por padrão. Use `--include-data`
somente se a branch precisar ser autocontida e se o licenciamento/distribuição
do FEMNIST estiver de acordo com o uso desejado; essa opção faz uma inclusão
forçada e consciente dos caches ignorados.

## Validação realizada

- `git fetch --prune origin`: comparação remota atualizada.
- `python3 -m compileall -q src tests`: passou.
- `git diff --check`: passou.
- `pytest`: não executado, pois o ambiente usado no levantamento não possui o
  pacote `pytest` instalado.

Antes da publicação definitiva, recomenda-se criar um ambiente virtual,
instalar as dependências e executar `python3 -m pytest -q`.

## Publicação segura em uma nova branch

O script publica no repositório pessoal `Sfgiovanni/UnL`, usando o remote
`personal`. O remote `origin` continua apontando para
`RenatoDSouza/QfederatedUlearning` e pode ser usado para comparação e
atualizações do projeto original.

O script exige um token com permissão de escrita em **Contents** no repositório
pessoal e nunca grava o token no remote ou no histórico:

```bash
export GITHUB_TOKEN='seu_token'
./scripts/publish_branch.sh \
  --branch feat/experimentos-unlearning \
  --message 'feat: aprimora experimentos de federated unlearning'
unset GITHUB_TOKEN
```

Para `Sfgiovanni/UnL`, pode ser usado um fine-grained PAT cujo `resource owner`
seja `Sfgiovanni`, com acesso ao repositório `UnL` e permissão **Contents:
Read and write**. Como alternativa, um PAT classic pode usar o escopo
`public_repo` (ou `repo`, se o repositório for privado).

Se o commit já tiver sido criado e somente o push tiver falhado, corrija o token
e retome sem recriar a branch ou o commit:

```bash
export GITHUB_TOKEN='novo_token_com_acesso'
./scripts/publish_branch.sh \
  --branch feat/experimentos-unlearning \
  --push-only
unset GITHUB_TOKEN
```

Para também versionar os dois caches locais, acrescente `--include-data`. O
push não usa `--force`; portanto, uma branch remota existente não será
sobrescrita.
