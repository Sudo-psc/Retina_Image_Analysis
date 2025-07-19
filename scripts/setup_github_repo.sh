#!/bin/bash

# Script para criar repositório GitHub para Retina Image Analysis
# Requer GitHub CLI (gh) instalado

set -e

echo "🚀 Configurando repositório GitHub para Retina Image Analysis"
echo "============================================================"

# Verificar se GitHub CLI está instalado
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) não está instalado."
    echo "📥 Instale com: brew install gh"
    echo "🔗 Ou baixe de: https://cli.github.com/"
    exit 1
fi

# Verificar se está logado no GitHub
if ! gh auth status &> /dev/null; then
    echo "🔐 Fazendo login no GitHub..."
    gh auth login
fi

# Informações do repositório
REPO_NAME="Retina_Image_Analysis"
REPO_DESCRIPTION="AI/ML project for retinal image analysis using Python with Software-Driven Development techniques"
REPO_VISIBILITY="public"  # ou "private"

echo "📋 Configurações do repositório:"
echo "   Nome: $REPO_NAME"
echo "   Descrição: $REPO_DESCRIPTION"
echo "   Visibilidade: $REPO_VISIBILITY"
echo ""

# Perguntar confirmação
read -p "🤔 Deseja continuar com essas configurações? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Operação cancelada."
    exit 1
fi

echo "📝 Preparando arquivos para commit..."

# Criar arquivo .env local se não existir
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Arquivo .env criado a partir do .env.example"
fi

# Adicionar todos os arquivos ao Git
echo "📁 Adicionando arquivos ao Git..."
git add .

# Fazer primeiro commit
echo "💾 Fazendo primeiro commit..."
git commit -m "feat: initial project setup

- Complete project structure with src/, data/, tests/, docs/
- Poetry configuration with ML/AI dependencies
- Docker and docker-compose setup
- CI/CD pipeline with GitHub Actions
- Pre-commit hooks configuration
- Comprehensive documentation (PT/EN)
- Data processing modules (RetinaDataset, ImagePreprocessor)
- Unit tests structure
- Configuration management

Project ready for Phase 2: Data Management"

# Criar repositório no GitHub
echo "🌐 Criando repositório no GitHub..."
gh repo create $REPO_NAME \
    --description "$REPO_DESCRIPTION" \
    --$REPO_VISIBILITY \
    --clone=false \
    --add-readme=false

# Adicionar remote origin
echo "🔗 Configurando remote origin..."
gh repo view $REPO_NAME --json sshUrl -q .sshUrl | xargs git remote add origin

# Push do código
echo "⬆️ Enviando código para GitHub..."
git push -u origin main

# Configurar branch protection (opcional)
echo "🛡️ Configurando proteção da branch main..."
gh api repos/:owner/$REPO_NAME/branches/main/protection \
    --method PUT \
    --field required_status_checks='{"strict":true,"contexts":["CI/CD Pipeline"]}' \
    --field enforce_admins=false \
    --field required_pull_request_reviews='{"required_approving_review_count":1,"dismiss_stale_reviews":true}' \
    --field restrictions=null \
    --field allow_force_pushes=false \
    --field allow_deletions=false 2>/dev/null || echo "⚠️ Não foi possível configurar proteção da branch (necessita permissões especiais)"

# Criar labels personalizadas
echo "🏷️ Criando labels do projeto..."
gh label create "phase-1" --description "Phase 1: Project Setup" --color "1d76db" --force
gh label create "phase-2" --description "Phase 2: Data Management" --color "0e8a16" --force
gh label create "phase-3" --description "Phase 3: Model Development" --color "fbca04" --force
gh label create "phase-4" --description "Phase 4: System Integration" --color "d93f0b" --force
gh label create "phase-5" --description "Phase 5: Deployment & Monitoring" --color "6f42c1" --force
gh label create "ml/data" --description "Machine Learning - Data related" --color "c2e0c6" --force
gh label create "ml/model" --description "Machine Learning - Model related" --color "ffeaa7" --force
gh label create "ml/training" --description "Machine Learning - Training related" --color "fd79a8" --force
gh label create "documentation" --description "Documentation improvements" --color "0075ca" --force
gh label create "infrastructure" --description "Infrastructure and DevOps" --color "e99695" --force

# Criar primeira issue
echo "📋 Criando primeira issue..."
gh issue create \
    --title "Phase 2: Implement Data Management Pipeline" \
    --body "## 🎯 Objetivo

Implementar o pipeline completo de gestão de dados para o projeto de análise de imagens de retina.

## 📋 Tarefas

### Download e Organização de Datasets
- [ ] Download do dataset DRIVE
- [ ] Download do dataset STARE  
- [ ] Download do dataset Messidor
- [ ] Download do dataset Kaggle Diabetic Retinopathy
- [ ] Organização dos dados conforme estrutura definida

### Pipeline de Processamento
- [ ] Completar módulo \`src/data/augmentation.py\`
- [ ] Implementar \`src/data/loader.py\` para DataLoaders
- [ ] Criar validação de qualidade de dados
- [ ] Implementar splits train/val/test automatizados

### Scripts de Automação
- [ ] Script de download automático de datasets
- [ ] Validação de integridade dos dados
- [ ] Estatísticas e análise exploratória automática

## 🎯 Critérios de Aceite

- [ ] Todos os datasets baixados e organizados
- [ ] Pipeline de preprocessamento funcional
- [ ] Data loaders implementados e testados
- [ ] Documentação atualizada
- [ ] Testes unitários para novos módulos

## 📚 Referências

- \`NEXT_STEPS.md\` - Roadmap detalhado
- \`data/raw/*/README.md\` - Instruções de download
- \`src/data/\` - Módulos de dados existentes

## 🔗 Links Úteis

- [DRIVE Dataset](https://drive.grand-challenge.org/)
- [STARE Dataset](http://cecas.clemson.edu/~ahoover/stare/)
- [Messidor Dataset](http://www.adcis.net/en/third-party/messidor/)
- [Kaggle DR Competition](https://www.kaggle.com/c/diabetic-retinopathy-detection)" \
    --label "phase-2,ml/data,enhancement" \
    --assignee @me

echo ""
echo "🎉 Repositório criado com sucesso!"
echo "======================================="
echo ""
echo "📊 Informações do repositório:"
gh repo view $REPO_NAME

echo ""
echo "🔗 Links úteis:"
echo "   📁 Repositório: $(gh repo view $REPO_NAME --json url -q .url)"
echo "   📋 Issues: $(gh repo view $REPO_NAME --json url -q .url)/issues"
echo "   🔄 Actions: $(gh repo view $REPO_NAME --json url -q .url)/actions"
echo "   📖 Wiki: $(gh repo view $REPO_NAME --json url -q .url)/wiki"
echo ""
echo "🚀 Próximos passos:"
echo "   1. Revisar a issue criada para Phase 2"
echo "   2. Fazer checkout da branch de desenvolvimento:"
echo "      git checkout -b feature/data-pipeline"
echo "   3. Começar implementação do pipeline de dados"
echo ""
echo "✅ Setup completo! Happy coding! 🚀"
