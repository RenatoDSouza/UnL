#!/usr/bin/env bash
set -euo pipefail

TARGET_REMOTE="personal"
TARGET_URL="https://github.com/Sfgiovanni/UnL.git"
EXPECTED_REPOSITORY="github.com/Sfgiovanni/UnL"
BRANCH=""
COMMIT_MESSAGE="docs: document and publish unlearning improvements"
INCLUDE_DATA=false
PUSH_ONLY=false

usage() {
  cat <<'EOF'
Uso:
  GITHUB_TOKEN=... ./scripts/publish_branch.sh [opcoes]

Destino: https://github.com/Sfgiovanni/UnL (remote "personal")

Opcoes:
  --branch NOME       Nome da nova branch. Padrao: feat/unlearning-update-DATA
  --message TEXTO     Mensagem do commit.
  --include-data      Inclui data/*.npz no commit (excluidos por padrao).
  --push-only         Retoma somente o push de uma branch local ja criada.
  -h, --help          Mostra esta ajuda.

O token precisa de permissao de escrita em Contents no repositorio.
Ele e fornecido ao Git somente via GIT_ASKPASS e nao e salvo no remote.
EOF
}

while (($#)); do
  case "$1" in
    --branch)
      [[ $# -ge 2 ]] || { echo "Erro: --branch requer um valor." >&2; exit 2; }
      BRANCH="$2"
      shift 2
      ;;
    --message)
      [[ $# -ge 2 ]] || { echo "Erro: --message requer um valor." >&2; exit 2; }
      COMMIT_MESSAGE="$2"
      shift 2
      ;;
    --include-data)
      INCLUDE_DATA=true
      shift
      ;;
    --push-only)
      PUSH_ONLY=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Erro: opcao desconhecida: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "Erro: defina GITHUB_TOKEN com um token de acesso ao GitHub." >&2
  exit 1
fi

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if ! git remote get-url "$TARGET_REMOTE" >/dev/null 2>&1; then
  git remote add "$TARGET_REMOTE" "$TARGET_URL"
fi

REMOTE_URL="$(git remote get-url "$TARGET_REMOTE")"
NORMALIZED_REMOTE="${REMOTE_URL%.git}"
NORMALIZED_REMOTE="${NORMALIZED_REMOTE#https://}"
NORMALIZED_REMOTE="${NORMALIZED_REMOTE#http://}"
NORMALIZED_REMOTE="${NORMALIZED_REMOTE#git@}"
NORMALIZED_REMOTE="${NORMALIZED_REMOTE/:/\/}"
if [[ "$NORMALIZED_REMOTE" != "$EXPECTED_REPOSITORY" ]]; then
  echo "Erro: o remote '$TARGET_REMOTE' aponta para '$REMOTE_URL', nao para o repositorio esperado." >&2
  exit 1
fi

if [[ -z "$BRANCH" ]]; then
  BRANCH="feat/unlearning-update-$(date +%Y%m%d-%H%M%S)"
fi
git check-ref-format --branch "$BRANCH" >/dev/null

if [[ "$PUSH_ONLY" == false ]] && git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  echo "Erro: a branch local '$BRANCH' ja existe." >&2
  exit 1
fi

if [[ "$PUSH_ONLY" == true ]]; then
  CURRENT_BRANCH="$(git branch --show-current)"
  if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
    echo "Erro: --push-only exige estar na branch '$BRANCH' (atual: '$CURRENT_BRANCH')." >&2
    exit 1
  fi
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "Erro: --push-only exige uma arvore de trabalho limpa." >&2
    exit 1
  fi
fi

if ! git config --get user.name >/dev/null || ! git config --get user.email >/dev/null; then
  echo "Erro: configure user.name e user.email no Git antes de publicar." >&2
  echo "Exemplo: git config user.name 'Seu Nome'" >&2
  echo "         git config user.email 'seu-email@example.com'" >&2
  exit 1
fi

ASKPASS="$(mktemp "${TMPDIR:-/tmp}/qfl-git-askpass.XXXXXX")"
cleanup() {
  rm -f "$ASKPASS"
  unset GITHUB_TOKEN
}
trap cleanup EXIT HUP INT TERM

cat >"$ASKPASS" <<'EOF'
#!/usr/bin/env bash
case "$1" in
  *Username*) printf '%s\n' 'x-access-token' ;;
  *Password*) printf '%s\n' "$GITHUB_TOKEN" ;;
  *) exit 1 ;;
esac
EOF
chmod 700 "$ASKPASS"

if [[ "$PUSH_ONLY" == false ]]; then
  set +e
  GIT_ASKPASS="$ASKPASS" \
  GIT_TERMINAL_PROMPT=0 \
  git ls-remote --exit-code --heads "$TARGET_REMOTE" "refs/heads/$BRANCH" >/dev/null
  REMOTE_BRANCH_STATUS=$?
  set -e
  case "$REMOTE_BRANCH_STATUS" in
    0)
      echo "Erro: a branch remota '$BRANCH' ja existe; escolha outro nome." >&2
      exit 1
      ;;
    2)
      ;;
    *)
      echo "Erro: nao foi possivel consultar o repositorio remoto; verifique o token e a rede." >&2
      exit 1
      ;;
  esac

  # O dry-run valida permissao de escrita antes de criar branch ou commit local.
  GIT_ASKPASS="$ASKPASS" \
  GIT_TERMINAL_PROMPT=0 \
  git push --dry-run "$TARGET_REMOTE" "HEAD:refs/heads/$BRANCH" >/dev/null || {
    echo "Erro: o token nao tem permissao para publicar no repositorio." >&2
    exit 1
  }

  git switch -c "$BRANCH"

  if [[ "$INCLUDE_DATA" == true ]]; then
    git add -A -- .
    shopt -s nullglob
    DATA_FILES=(data/*.npz)
    shopt -u nullglob
    if ((${#DATA_FILES[@]})); then
      git add -f -- "${DATA_FILES[@]}"
    else
      echo "Aviso: --include-data foi usado, mas nenhum data/*.npz foi encontrado." >&2
    fi
  else
    git add -A -- .
  fi

  if git diff --cached --quiet; then
    echo "Erro: nao ha mudancas para criar o commit." >&2
    exit 1
  fi

  git commit -m "$COMMIT_MESSAGE"
fi

GIT_ASKPASS="$ASKPASS" \
GIT_TERMINAL_PROMPT=0 \
git push --set-upstream "$TARGET_REMOTE" "$BRANCH"

echo "Branch publicada: $BRANCH"
echo "URL: https://github.com/Sfgiovanni/UnL/tree/$BRANCH"
